from __future__ import annotations

import shlex
from collections import Counter
from typing import Any


def criterion_index(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        criterion["id"]: criterion
        for obligation in ledger["obligations"]
        for criterion in obligation["acceptance_criteria"]
    }


def validate_issue_source_anchors(
    sources: list[dict[str, Any]], *, title: str, body: str
) -> list[str]:
    errors: list[str] = []
    fields = {"title": title, "body": body}
    for source in sources:
        if source.get("kind") != "issue":
            continue
        locator = source.get("locator") or {}
        field = locator.get("field")
        quote = locator.get("quote")
        if field not in fields or not isinstance(quote, str) or quote not in fields[field]:
            errors.append(f"{source.get('id', 'source')} issue quote is not exact in {field}")
    return errors


def validate_ledger(
    payload: dict[str, Any],
    *,
    task_id: str,
    source_policy_sha256: str | None,
) -> list[str]:
    errors: list[str] = []
    if payload["task_id"] != task_id:
        errors.append(f"task_id {payload['task_id']!r} does not match {task_id!r}")
    if source_policy_sha256 and payload["source_policy_sha256"] != source_policy_sha256:
        errors.append("source_policy_sha256 does not match the frozen source policy")

    source_ids = [source["id"] for source in payload["sources"]]
    _append_duplicate_error(errors, source_ids, "source IDs")
    known_sources = set(source_ids)

    obligations = payload["obligations"]
    obligation_ids = [item["id"] for item in obligations]
    _append_duplicate_error(errors, obligation_ids, "obligation IDs")
    known_obligations = set(obligation_ids)
    criteria: list[dict[str, Any]] = []
    graph: dict[str, tuple[str, ...]] = {}
    for obligation in obligations:
        obligation_id = obligation["id"]
        graph[obligation_id] = tuple(obligation["dependencies"])
        for dependency in obligation["dependencies"]:
            if dependency not in known_obligations:
                errors.append(f"{obligation_id} has dangling dependency {dependency}")
            if dependency == obligation_id:
                errors.append(f"{obligation_id} cannot depend on itself")
        for criterion in obligation["acceptance_criteria"]:
            criteria.append(criterion)
            if not criterion["id"].startswith(f"{obligation_id}."):
                errors.append(f"criterion {criterion['id']} does not belong to {obligation_id}")
    cycle = _find_cycle(graph)
    if cycle:
        errors.append("dependency graph contains cycle: " + " -> ".join(cycle))

    criterion_ids = [item["id"] for item in criteria]
    _append_duplicate_error(errors, criterion_ids, "criterion IDs")
    for criterion in criteria:
        ref = criterion["id"]
        _validate_source_refs(errors, ref, criterion["source_refs"], known_sources)
        unspecified = criterion["basis_kind"] == "unspecified"
        if unspecified != (criterion["criticality"] == "unspecified"):
            errors.append(
                f"{ref} basis_kind and criticality must both be unspecified or both binding"
            )
        source_kinds = {
            source["kind"]
            for source in payload["sources"]
            if source["id"] in criterion["source_refs"]
        }
        if criterion["basis_kind"] == "explicit_issue" and "issue" not in source_kinds:
            errors.append(f"{ref} explicit_issue basis requires an issue source")
        if (
            criterion["basis_kind"]
            in {
                "repository_contract",
                "compatibility_inferred",
            }
            and "repository" not in source_kinds
        ):
            errors.append(f"{ref} repository basis requires a repository source")
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
    protocol_errors: list[str] = []
    if payload["task_id"] != task_id:
        protocol_errors.append("assessment task_id does not match Controller input")
    if payload["patch_sha256"] != patch_sha256:
        protocol_errors.append("assessment patch_sha256 does not match P0")
    if payload["ledger_sha256"] != ledger_sha256:
        protocol_errors.append("assessment ledger_sha256 does not match frozen ledger")

    ledger_criteria = criterion_index(ledger)
    assessments = payload["criterion_assessments"]
    actual_refs = [item["criterion_ref"] for item in assessments]
    _append_duplicate_error(errors, actual_refs, "criterion assessments")
    missing = sorted(set(ledger_criteria) - set(actual_refs))
    extra = sorted(set(actual_refs) - set(ledger_criteria))
    if missing:
        errors.append("missing criterion assessments: " + ", ".join(missing))
    if extra:
        errors.append("unknown criterion assessments: " + ", ".join(extra))

    sources = [*ledger["sources"], *payload["additional_sources"]]
    source_ids = [source["id"] for source in sources]
    _append_duplicate_error(errors, source_ids, "combined source IDs")
    known_sources = set(source_ids)

    challenges = payload["ledger_challenges"]
    challenge_ids = [item["id"] for item in challenges]
    _append_duplicate_error(errors, challenge_ids, "challenge IDs")
    challenges_by_ref: dict[str, list[dict[str, Any]]] = {}
    late = {item["id"]: item for item in payload["late_obligation_candidates"]}
    if len(late) != len(payload["late_obligation_candidates"]):
        errors.append("late criterion IDs must be unique")

    for challenge in challenges:
        ref = challenge["criterion_ref"]
        if ref not in ledger_criteria:
            errors.append(f"challenge {challenge['id']} references unknown criterion {ref}")
        challenges_by_ref.setdefault(ref, []).append(challenge)
        _validate_source_refs(errors, challenge["id"], challenge["source_refs"], known_sources)
        original_sources = set(ledger_criteria.get(ref, {}).get("source_refs", []))
        missing_original_sources = sorted(original_sources - set(challenge["source_refs"]))
        if missing_original_sources:
            errors.append(
                f"challenge {challenge['id']} omits frozen criterion sources: "
                + ", ".join(missing_original_sources)
            )
        replacements = challenge["replacement_refs"]
        unknown_replacements = sorted(set(replacements) - set(late))
        if unknown_replacements:
            errors.append(
                f"challenge {challenge['id']} has unknown replacements: "
                + ", ".join(unknown_replacements)
            )
        if challenge["disposition"] == "replaced" and not replacements:
            errors.append(f"challenge {challenge['id']} replaced disposition needs replacements")
        if challenge["disposition"] != "replaced" and replacements:
            errors.append(
                f"challenge {challenge['id']} has replacements without replaced disposition"
            )

    for item in assessments:
        ref = item["criterion_ref"]
        source_check = item["source_check"]
        _validate_source_refs(errors, ref, source_check["source_refs"], known_sources)
        original_sources = set(ledger_criteria.get(ref, {}).get("source_refs", []))
        missing_original_sources = sorted(original_sources - set(source_check["source_refs"]))
        if missing_original_sources:
            errors.append(
                f"{ref} source check omits frozen criterion sources: "
                + ", ".join(missing_original_sources)
            )
        status = source_check["status"]
        linked = challenges_by_ref.get(ref, [])
        if status == "supported" and linked:
            errors.append(f"{ref} is supported but also has a ledger challenge")
        if status != "supported" and len(linked) != 1:
            errors.append(f"{ref} source status {status} requires exactly one challenge")
        if linked and not _challenge_matches_source_status(linked[0], status):
            errors.append(f"{ref} challenge kind/disposition conflicts with source status {status}")
        implementation_status = item["implementation_status"]
        if implementation_status == "not_applicable":
            if not linked or linked[0]["disposition"] not in {"nonbinding", "replaced"}:
                errors.append(f"{ref} not_applicable requires nonbinding or replaced challenge")
        elif (
            status != "supported"
            and linked
            and linked[0]["disposition"]
            in {
                "nonbinding",
                "replaced",
            }
        ):
            errors.append(f"{ref} nonbinding/replaced criterion must be not_applicable")
        _validate_implementation_record(errors, ref, item)

    for ref, item in late.items():
        _validate_source_refs(errors, ref, item["source_refs"], known_sources)
        _validate_source_refs(errors, ref, item["source_check"]["source_refs"], known_sources)
        missing_late_sources = sorted(
            set(item["source_refs"]) - set(item["source_check"]["source_refs"])
        )
        if missing_late_sources:
            errors.append(
                f"{ref} source check omits late criterion sources: "
                + ", ".join(missing_late_sources)
            )
        unspecified = item["basis_kind"] == "unspecified"
        if unspecified != (item["criticality"] == "unspecified"):
            errors.append(f"{ref} basis_kind and criticality must align on unspecified")
        if item["source_check"]["status"] != "supported":
            errors.append(f"{ref} late criterion must have a supported source check")
        if item["implementation_status"] == "not_applicable":
            errors.append(f"{ref} late criterion cannot be not_applicable")
        source_kinds = {source["kind"] for source in sources if source["id"] in item["source_refs"]}
        if item["basis_kind"] == "explicit_issue" and "issue" not in source_kinds:
            errors.append(f"{ref} explicit_issue basis requires an issue source")
        if (
            item["basis_kind"]
            in {
                "repository_contract",
                "compatibility_inferred",
            }
            and "repository" not in source_kinds
        ):
            errors.append(f"{ref} repository basis requires a repository source")
        _validate_implementation_record(errors, ref, item)

    effective = _effective_criterion_records(ledger, payload)
    effective_by_ref = {item["criterion_ref"]: item for item in effective}
    regression_by_id = {item["id"]: item for item in payload["unexpected_regressions"]}
    known_findings = {
        *ledger_criteria,
        *late,
        *regression_by_id,
    }
    requests = payload["repair_requests"]
    request_ids = [item["id"] for item in requests]
    _append_duplicate_error(errors, request_ids, "repair request IDs")
    priorities = [item["priority"] for item in requests]
    if priorities != list(range(1, len(priorities) + 1)):
        errors.append("repair request priorities must be ordered and continuous from 1")
    for request in requests:
        _validate_source_refs(errors, request["id"], request["source_refs"], known_sources)
        finding_refs = set(request["finding_refs"])
        criterion_refs = set(request["criterion_refs"])
        unknown_findings = sorted(finding_refs - known_findings)
        if unknown_findings:
            errors.append(
                f"repair request {request['id']} references unknown findings: "
                + ", ".join(unknown_findings)
            )
        criterion_findings = {ref for ref in finding_refs if not ref.startswith("R")}
        non_effective_findings = sorted(criterion_findings - set(effective_by_ref))
        if non_effective_findings:
            errors.append(
                f"repair request {request['id']} uses non-effective criterion findings: "
                + ", ".join(non_effective_findings)
            )
        unknown_criteria = sorted(criterion_refs - set(effective_by_ref))
        if unknown_criteria:
            errors.append(
                f"repair request {request['id']} references non-effective criteria: "
                + ", ".join(unknown_criteria)
            )
        if criterion_findings != criterion_refs:
            errors.append(
                f"repair request {request['id']} criterion_refs must exactly match its "
                "criterion finding_refs"
            )
        if not request["criterion_refs"] and not any(
            ref.startswith("R") for ref in request["finding_refs"]
        ):
            errors.append(f"repair request {request['id']} has no effective criterion")
        required_sources: set[str] = set()
        for ref in criterion_refs:
            required_sources.update(effective_by_ref.get(ref, {}).get("source_refs", []))
        for ref in finding_refs:
            required_sources.update(regression_by_id.get(ref, {}).get("source_refs", []))
        missing_sources = sorted(required_sources - set(request["source_refs"]))
        if missing_sources:
            errors.append(
                f"repair request {request['id']} omits finding sources: "
                + ", ".join(missing_sources)
            )

        statuses = {
            ref: effective_by_ref.get(ref, {}).get("implementation_status")
            for ref in criterion_refs
        }
        action_kind = request["action_kind"]
        if action_kind == "fix":
            uncertain = sorted(ref for ref, status in statuses.items() if status == "uncertain")
            if uncertain:
                errors.append(
                    f"repair request {request['id']} must investigate uncertain criteria: "
                    + ", ".join(uncertain)
                )
            no_confirmed_gap = sorted(
                ref
                for ref, status in statuses.items()
                if status not in {"partial", "missing", "uncertain"}
            )
            if no_confirmed_gap:
                errors.append(
                    f"repair request {request['id']} cannot fix criteria without a confirmed "
                    "partial/missing gap: " + ", ".join(no_confirmed_gap)
                )
        elif criterion_refs:
            invalid_investigation = sorted(
                ref for ref, status in statuses.items() if status != "uncertain"
            )
            if invalid_investigation:
                errors.append(
                    f"repair request {request['id']} can investigate only uncertain criteria: "
                    + ", ".join(invalid_investigation)
                )
            if any(ref.startswith("R") for ref in finding_refs):
                errors.append(
                    f"repair request {request['id']} cannot investigate an already evidenced "
                    "unexpected regression"
                )
        elif any(ref.startswith("R") for ref in finding_refs):
            errors.append(
                f"repair request {request['id']} cannot investigate an already evidenced "
                "unexpected regression"
            )

    regression_ids = [item["id"] for item in payload["unexpected_regressions"]]
    _append_duplicate_error(errors, regression_ids, "unexpected regression IDs")
    for regression in payload["unexpected_regressions"]:
        _validate_source_refs(errors, regression["id"], regression["source_refs"], known_sources)
        _validate_evidence_lines(errors, regression["id"], regression["implementation_evidence"])
        if regression["severity"] == "blocker" and not regression["blocking"]:
            errors.append(f"{regression['id']} blocker regression must set blocking=true")

    must_gaps = [
        item["criterion_ref"]
        for item in effective
        if item["criticality"] == "must" and item["implementation_status"] != "covered"
    ]
    unresolved_must = [
        ref
        for ref, linked in challenges_by_ref.items()
        if linked[0]["disposition"] == "unresolved"
        and ledger_criteria.get(ref, {}).get("criticality") == "must"
    ]
    blocking_regression = any(item["blocking"] for item in payload["unexpected_regressions"])
    verdict = payload["verdict"]
    if verdict == "approve":
        if must_gaps:
            errors.append(
                "approve conflicts with uncovered effective must criteria: " + ", ".join(must_gaps)
            )
        if unresolved_must:
            errors.append(
                "approve conflicts with unresolved must criteria: " + ", ".join(unresolved_must)
            )
        if blocking_regression:
            errors.append("approve conflicts with a blocking unexpected regression")
        if requests:
            errors.append("approve verdict cannot include repair requests")
    elif verdict == "abstain":
        if requests:
            errors.append("abstain verdict cannot include repair requests")
        if not _nonempty(payload["abstain_reason"]):
            errors.append("abstain verdict requires abstain_reason")
        uncertain_must = [
            item["criterion_ref"]
            for item in effective
            if item["criticality"] == "must" and item["implementation_status"] == "uncertain"
        ]
        if not unresolved_must and not uncertain_must:
            errors.append(
                "abstain requires an unresolved or implementation-uncertain must criterion"
            )
    else:
        if not requests:
            errors.append(f"{verdict} verdict requires at least one repair request")
        if payload["abstain_reason"] is not None:
            errors.append(f"{verdict} verdict cannot include abstain_reason")
    if verdict != "abstain" and payload["abstain_reason"] is not None:
        errors.append(f"{verdict} verdict cannot include abstain_reason")
    return errors, protocol_errors


def build_effective_review_set(
    *,
    task_id: str,
    ledger_sha256: str,
    assessment_sha256: str,
    assessment: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    challenges = {item["criterion_ref"]: item for item in assessment["ledger_challenges"]}
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "ledger_sha256": ledger_sha256,
        "assessment_sha256": assessment_sha256,
        "effective_criteria": _effective_criterion_records(ledger, assessment),
        "nonbinding_criteria": sorted(
            ref for ref, item in challenges.items() if item["disposition"] == "nonbinding"
        ),
        "replaced_criteria": sorted(
            (
                {"criterion_ref": ref, "replacement_refs": item["replacement_refs"]}
                for ref, item in challenges.items()
                if item["disposition"] == "replaced"
            ),
            key=lambda item: item["criterion_ref"],
        ),
        "unresolved_criteria": sorted(
            ref for ref, item in challenges.items() if item["disposition"] == "unresolved"
        ),
        "repair_request_ids": [item["id"] for item in assessment["repair_requests"]],
        "verdict": assessment["verdict"],
    }


def validate_repair_audit(
    payload: dict[str, Any],
    *,
    task_id: str,
    p0_sha256: str,
    assessment_sha256: str,
    effective_review_set_sha256: str,
    ledger: dict[str, Any],
    assessment: dict[str, Any],
    effective_review_set: dict[str, Any],
    actual_changed_paths: set[str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    protocol: list[str] = []
    bindings = {
        "task_id": task_id,
        "p0_sha256": p0_sha256,
        "assessment_sha256": assessment_sha256,
        "effective_review_set_sha256": effective_review_set_sha256,
    }
    for field, expected in bindings.items():
        if payload[field] != expected:
            protocol.append(f"repair audit {field} does not match frozen input")
    declared_paths = set(payload["changed_paths"])
    if declared_paths != actual_changed_paths:
        errors.append(
            "repair audit changed_paths do not match actual P0-to-P1 diff: "
            f"declared={sorted(declared_paths)!r}, actual={sorted(actual_changed_paths)!r}"
        )

    all_sources = [
        *ledger["sources"],
        *assessment["additional_sources"],
        *payload["additional_sources"],
    ]
    source_ids = [item["id"] for item in all_sources]
    _append_duplicate_error(errors, source_ids, "repair audit combined source IDs")
    known_sources = set(source_ids)
    known_requests = {item["id"] for item in assessment["repair_requests"]}
    requests_by_id = {item["id"]: item for item in assessment["repair_requests"]}
    known_criteria = {item["criterion_ref"] for item in effective_review_set["effective_criteria"]}
    effective_by_ref = {
        item["criterion_ref"]: item for item in effective_review_set["effective_criteria"]
    }
    change_ids = [item["change_id"] for item in payload["changes"]]
    _append_duplicate_error(errors, change_ids, "repair audit change IDs")
    covered_requests: set[str] = set()
    covered_request_criteria: dict[str, set[str]] = {}
    audited_changed_paths: set[str] = set()
    for change in payload["changes"]:
        unknown_requests = sorted(set(change["request_refs"]) - known_requests)
        if unknown_requests:
            errors.append(
                f"{change['change_id']} references unknown requests: " + ", ".join(unknown_requests)
            )
        covered_requests.update(change["request_refs"])
        for request_ref in change["request_refs"]:
            covered_request_criteria.setdefault(request_ref, set()).update(change["criterion_refs"])
        unknown_criteria = sorted(set(change["criterion_refs"]) - known_criteria)
        if unknown_criteria:
            errors.append(
                f"{change['change_id']} references non-effective criteria: "
                + ", ".join(unknown_criteria)
            )
        requested_criteria: set[str] = set()
        required_sources: set[str] = set()
        for request_ref in change["request_refs"]:
            request = requests_by_id.get(request_ref)
            if request is None:
                continue
            requested_criteria.update(request["criterion_refs"])
            required_sources.update(request["source_refs"])
        change_criteria = set(change["criterion_refs"])
        for criterion_ref in change_criteria:
            required_sources.update(effective_by_ref.get(criterion_ref, {}).get("source_refs", []))
        extra_criteria = sorted(change_criteria - requested_criteria)
        if (
            change["request_refs"]
            and extra_criteria
            and "out_of_request_scope" not in change["risk_tags"]
        ):
            errors.append(
                f"{change['change_id']} cites criteria outside its requests without the "
                "out_of_request_scope risk tag: " + ", ".join(extra_criteria)
            )
        _validate_source_refs(errors, change["change_id"], change["source_refs"], known_sources)
        missing_sources = sorted(required_sources - set(change["source_refs"]))
        if missing_sources:
            errors.append(
                f"{change['change_id']} omits request/criterion sources: "
                + ", ".join(missing_sources)
            )
        decision = change["decision"]
        source_check = change["source_check"]
        allowed = {
            "implement": {"supported", "design_choice"},
            "defer": {"unresolved"},
            "reject": {"contradicted"},
        }
        if source_check not in allowed[decision]:
            errors.append(
                f"{change['change_id']} decision {decision} conflicts with source_check {source_check}"
            )
        changed = set(change["changed_paths"])
        if decision == "implement" and not changed:
            errors.append(f"{change['change_id']} implemented change has no changed_paths")
        if decision == "implement" and not change["validation"]:
            errors.append(f"{change['change_id']} implemented change has no validation record")
        if decision != "implement" and changed:
            errors.append(f"{change['change_id']} deferred/rejected change has changed_paths")
        if not changed.issubset(actual_changed_paths):
            errors.append(f"{change['change_id']} cites paths outside the actual diff")
        if decision == "implement":
            audited_changed_paths.update(changed)
        if not change["request_refs"] and "out_of_request_scope" not in change["risk_tags"]:
            errors.append(
                f"{change['change_id']} has no request_refs and must be tagged out_of_request_scope"
            )
        if source_check == "design_choice" and not (
            change["request_refs"] or change["criterion_refs"]
        ):
            errors.append(
                f"{change['change_id']} design_choice is not tied to a request or effective "
                "criterion"
            )

    unaddressed_ids = [item["request_ref"] for item in payload["unaddressed_requests"]]
    _append_duplicate_error(errors, unaddressed_ids, "unaddressed request refs")
    unknown_unaddressed = sorted(set(unaddressed_ids) - known_requests)
    if unknown_unaddressed:
        errors.append("unknown unaddressed requests: " + ", ".join(unknown_unaddressed))
    overlap = sorted(covered_requests & set(unaddressed_ids))
    if overlap:
        errors.append("requests cannot be both addressed and unaddressed: " + ", ".join(overlap))
    uncovered = sorted(known_requests - covered_requests - set(unaddressed_ids))
    if uncovered:
        errors.append("repair audit omits requests: " + ", ".join(uncovered))
    for request_ref in sorted(covered_requests - set(unaddressed_ids)):
        request = requests_by_id.get(request_ref)
        if request is None:
            continue
        missing_criteria = sorted(
            set(request["criterion_refs"]) - covered_request_criteria.get(request_ref, set())
        )
        if missing_criteria:
            errors.append(
                f"repair audit {request_ref} omits its requested criteria: "
                + ", ".join(missing_criteria)
            )
    unaudited_paths = sorted(actual_changed_paths - audited_changed_paths)
    if unaudited_paths:
        errors.append(
            "repair audit has actual changed paths without an implemented change record: "
            + ", ".join(unaudited_paths)
        )
    return errors, protocol


def patch_changed_paths(patch: bytes) -> set[str]:
    paths: set[str] = set()
    for raw_line in patch.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith("diff --git "):
            continue
        try:
            parts = shlex.split(raw_line)
        except ValueError:
            continue
        if len(parts) >= 4:
            path = parts[3]
            paths.add(path.removeprefix("b/"))
    return paths


def changed_paths_between_patches(before: bytes, after: bytes) -> set[str]:
    before_files = _patch_file_sections(before)
    after_files = _patch_file_sections(after)
    return {
        path
        for path in before_files.keys() | after_files.keys()
        if before_files.get(path) != after_files.get(path)
    }


def source_check_counts(assessment: dict[str, Any]) -> Counter[str]:
    result: Counter[str] = Counter()
    for item in assessment.get("criterion_assessments") or []:
        result[item["source_check"]["status"]] += 1
    for item in assessment.get("late_obligation_candidates") or []:
        result[item["source_check"]["status"]] += 1
    return result


def _effective_criterion_records(
    ledger: dict[str, Any], assessment: dict[str, Any]
) -> list[dict[str, Any]]:
    original = criterion_index(ledger)
    assessment_by_ref = {
        item["criterion_ref"]: item for item in assessment["criterion_assessments"]
    }
    challenges = {item["criterion_ref"]: item for item in assessment["ledger_challenges"]}
    result: list[dict[str, Any]] = []
    for ref, criterion in original.items():
        item = assessment_by_ref.get(ref)
        if item is None or criterion["criticality"] == "unspecified":
            continue
        challenge = challenges.get(ref)
        if item["source_check"]["status"] != "supported" or challenge is not None:
            continue
        result.append(
            {
                "criterion_ref": ref,
                "origin": "ledger",
                "criticality": criterion["criticality"],
                "source_refs": item["source_check"]["source_refs"],
                "implementation_status": item["implementation_status"],
            }
        )
    for item in assessment["late_obligation_candidates"]:
        if item["criticality"] == "unspecified" or item["source_check"]["status"] != "supported":
            continue
        result.append(
            {
                "criterion_ref": item["id"],
                "origin": "late",
                "criticality": item["criticality"],
                "source_refs": item["source_check"]["source_refs"],
                "implementation_status": item["implementation_status"],
            }
        )
    return sorted(result, key=lambda item: item["criterion_ref"])


def _patch_file_sections(patch: bytes) -> dict[str, bytes]:
    sections: dict[str, bytes] = {}
    current_path: str | None = None
    current: list[bytes] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith(b"diff --git "):
            if current_path is not None:
                sections[current_path] = b"".join(current)
            current = [line]
            try:
                parts = shlex.split(line.decode("utf-8", errors="replace"))
            except ValueError:
                current_path = None
            else:
                raw_path = parts[3] if len(parts) >= 4 else ""
                current_path = raw_path.removeprefix("b/")
        elif current_path is not None:
            current.append(line)
    if current_path is not None:
        sections[current_path] = b"".join(current)
    return sections


def _validate_implementation_record(errors: list[str], ref: str, item: dict[str, Any]) -> None:
    status = item["implementation_status"]
    evidence = item["implementation_evidence"]
    tests = item["test_evidence"]
    note = item.get("validation_note")
    gap = item.get("gap")
    if status == "covered":
        if not evidence:
            errors.append(f"{ref} is covered without implementation evidence")
        if not tests and not _nonempty(note):
            errors.append(f"{ref} is covered without test evidence or validation_note")
        nonpassing_tests = [item["test_name"] for item in tests if item["result"] != "passed"]
        if nonpassing_tests:
            errors.append(
                f"{ref} covered evidence contains non-passing tests: "
                + ", ".join(nonpassing_tests)
            )
        if _nonempty(gap):
            errors.append(f"{ref} is covered but has a gap")
    elif status != "not_applicable" and not _nonempty(gap):
        errors.append(f"{ref} implementation status {status} requires a non-empty gap")
    _validate_evidence_lines(errors, ref, evidence)


def _validate_evidence_lines(
    errors: list[str], ref: str, evidence_items: list[dict[str, Any]]
) -> None:
    for evidence in evidence_items:
        start, end = evidence["line_start"], evidence["line_end"]
        if (start is None) != (end is None):
            errors.append(f"{ref} evidence line_start and line_end must both be set")
        elif start is not None and end < start:
            errors.append(f"{ref} evidence line_end precedes line_start")


def _challenge_matches_source_status(challenge: dict[str, Any], status: str) -> bool:
    expected_kind = {
        "overstated": "unsupported_specificity",
        "understated": "weakened_requirement",
        "conflicting": "source_conflict",
        "underdetermined": "unresolved_meaning",
    }
    if status == "supported" or challenge["kind"] != expected_kind.get(status):
        return False
    if status == "understated":
        return challenge["disposition"] == "replaced"
    if status == "conflicting":
        return challenge["disposition"] == "unresolved"
    return True


def _append_duplicate_error(errors: list[str], values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        errors.append(f"{label} must be unique")


def _validate_source_refs(
    errors: list[str], owner: str, refs: list[str], known_sources: set[str]
) -> None:
    unknown = sorted(set(refs) - known_sources)
    if unknown:
        errors.append(f"{owner} references unknown sources: " + ", ".join(unknown))


def _find_cycle(graph: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    active: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> tuple[str, ...] | None:
        if node in active:
            start = active.index(node)
            return tuple(active[start:] + [node])
        if node in visited:
            return None
        active.append(node)
        for dependency in graph.get(node, ()):
            if dependency in graph:
                cycle = visit(dependency)
                if cycle:
                    return cycle
        active.pop()
        visited.add(node)
        return None

    for node in graph:
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
