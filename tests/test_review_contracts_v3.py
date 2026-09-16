import copy

import pytest

from external_agents_eval.workflows.solve_review_repair.config import DEFAULT_SCHEMA_DIR
from external_agents_eval.workflows.solve_review_repair.validators import (
    ContractValidationError,
    ContractValidator,
)

PATCH_SHA = "2" * 64
LEDGER_SHA = "3" * 64


def ledger_v3():
    return {
        "schema_version": "3.0",
        "task_id": "task-a",
        "summary": "Expose and apply a timeout while preserving the public default.",
        "sources": [
            {
                "id": "S01",
                "kind": "task",
                "field": "description",
                "quote": "Accept a timeout.",
            },
            {
                "id": "S02",
                "kind": "repository",
                "path": "src/client.py",
                "line_start": 1,
                "line_end": 4,
            },
        ],
        "obligations": [
            {
                "id": "O01",
                "statement": "The public client accepts a timeout.",
                "criticality": "must",
                "basis": "explicit_task",
                "source_refs": ["S01"],
                "rationale": "The task explicitly requests the input.",
            },
            {
                "id": "O02",
                "statement": "Calls that omit the timeout keep the documented default.",
                "criticality": "should",
                "basis": "repository_contract",
                "source_refs": ["S02"],
                "rationale": "The public constructor documents the default.",
            },
        ],
        "open_questions": [
            {
                "id": "U01",
                "question": "The task does not prescribe an internal clock type.",
                "source_refs": ["S01", "S02"],
            }
        ],
    }


def implementation_evidence():
    return [
        {
            "path": "src/client.py",
            "line_start": 5,
            "line_end": 9,
            "explanation": "The public constructor stores and applies the timeout.",
        }
    ]


def assessment_item(ref, *, implementation_status="covered", source_status="supported"):
    return {
        "obligation_ref": ref,
        "source_status": source_status,
        "source_note": None if source_status == "supported" else "The source is ambiguous.",
        "implementation_status": (
            implementation_status if source_status == "supported" else "not_applicable"
        ),
        "implementation_evidence": (
            implementation_evidence()
            if source_status == "supported" and implementation_status != "missing"
            else []
        ),
        "validation_evidence": [],
        "gap": (
            None
            if implementation_status == "covered" or source_status != "supported"
            else "The behavior is not established."
        ),
    }


def assessment_v3():
    return {
        "schema_version": "3.0",
        "task_id": "task-a",
        "patch_sha256": PATCH_SHA,
        "ledger_sha256": LEDGER_SHA,
        "additional_sources": [],
        "late_obligations": [],
        "obligation_assessments": [assessment_item("O01"), assessment_item("O02")],
        "unexpected_regressions": [],
        "repair_requests": [],
        "verdict": "approve",
        "verdict_rationale": "Every supported must obligation is covered.",
    }


@pytest.fixture
def validator_v3():
    return ContractValidator(DEFAULT_SCHEMA_DIR, contract_version="3.0")


def validate_assessment(validator, payload, ledger=None):
    validator.validate_assessment(
        payload,
        task_id="task-a",
        patch_sha256=PATCH_SHA,
        ledger_sha256=LEDGER_SHA,
        ledger=ledger or ledger_v3(),
    )


def test_v3_schema_bundle_and_minimal_ledger_are_valid(validator_v3):
    assert set(validator_v3.SCHEMAS) == {
        "build_input",
        "ledger",
        "source_bindings",
        "assessment_input",
        "assessment",
        "effective_review_set",
        "repair_input",
        "repair_audit",
    }
    minimal = {
        "schema_version": "3.0",
        "task_id": "task-a",
        "summary": "Accept a timeout.",
        "sources": [
            {
                "id": "S01",
                "kind": "task",
                "field": "description",
                "quote": "Accept a timeout.",
            }
        ],
        "obligations": [
            {
                "id": "O01",
                "statement": "The public client accepts a timeout.",
                "criticality": "must",
                "basis": "explicit_task",
                "source_refs": ["S01"],
                "rationale": "The task explicitly requests the input.",
            }
        ],
        "open_questions": [],
    }
    validator_v3.validate_ledger(minimal, task_id="task-a")
    validator_v3.validate_ledger(ledger_v3(), task_id="task-a")
    validator_v3.validate_task_source_anchors(
        ledger_v3()["sources"],
        title="Timeout",
        description="Accept a timeout.",
    )


def test_v3_open_questions_do_not_become_obligations(validator_v3):
    payload = ledger_v3()
    assert {item["id"] for item in payload["obligations"]} == {"O01", "O02"}
    assert payload["open_questions"][0]["id"] == "U01"
    validator_v3.validate_ledger(payload, task_id="task-a")


def test_v3_basis_requires_the_correct_source_kind(validator_v3):
    payload = ledger_v3()
    payload["obligations"][0]["basis"] = "repository_contract"
    with pytest.raises(ContractValidationError, match="repository basis requires"):
        validator_v3.validate_ledger(payload, task_id="task-a")


def test_v3_every_original_and_late_obligation_is_assessed_once(validator_v3):
    payload = assessment_v3()
    payload["additional_sources"] = [
        {
            "id": "A01",
            "kind": "task",
            "field": "description",
            "quote": "Accept a timeout.",
        }
    ]
    payload["late_obligations"] = [
        {
            "id": "L01",
            "statement": "Retries also apply the timeout.",
            "criticality": "must",
            "basis": "explicit_task",
            "source_refs": ["A01"],
            "rationale": "The task applies the requirement to retries.",
        }
    ]
    payload["obligation_assessments"].append(assessment_item("L01"))
    validate_assessment(validator_v3, payload)

    missing = copy.deepcopy(payload)
    missing["obligation_assessments"].pop()
    with pytest.raises(ContractValidationError, match="missing obligation assessments: L01"):
        validate_assessment(validator_v3, missing)

    duplicate = copy.deepcopy(payload)
    duplicate["obligation_assessments"].append(copy.deepcopy(assessment_item("L01")))
    with pytest.raises(ContractValidationError, match="obligation assessments must be unique"):
        validate_assessment(validator_v3, duplicate)


def test_v3_additional_sources_use_a_ids_and_late_obligations_use_l_ids(validator_v3):
    wrong_source = assessment_v3()
    wrong_source["additional_sources"] = [
        {
            "id": "S03",
            "kind": "task",
            "field": "description",
            "quote": "Accept a timeout.",
        }
    ]
    with pytest.raises(ContractValidationError, match="does not match"):
        validate_assessment(validator_v3, wrong_source)

    wrong_late = assessment_v3()
    wrong_late["late_obligations"] = [
        {
            "id": "O03",
            "statement": "Retries apply the timeout.",
            "criticality": "must",
            "basis": "explicit_task",
            "source_refs": ["S01"],
            "rationale": "The task establishes the retry behavior.",
        }
    ]
    with pytest.raises(ContractValidationError, match="does not match"):
        validate_assessment(validator_v3, wrong_late)


def test_v3_source_and_implementation_statuses_are_cross_checked(validator_v3):
    unsupported = assessment_v3()
    unsupported["obligation_assessments"][0] = assessment_item(
        "O01", source_status="unsupported"
    )
    validate_assessment(validator_v3, unsupported)

    invalid = copy.deepcopy(unsupported)
    invalid["obligation_assessments"][0]["implementation_status"] = "missing"
    with pytest.raises(ContractValidationError, match="must be not_applicable"):
        validate_assessment(validator_v3, invalid)


def test_v3_covered_requires_implementation_evidence_and_only_passing_validation(
    validator_v3,
):
    no_evidence = assessment_v3()
    no_evidence["obligation_assessments"][0]["implementation_evidence"] = []
    with pytest.raises(ContractValidationError, match="covered without implementation"):
        validate_assessment(validator_v3, no_evidence)

    failed = assessment_v3()
    failed["obligation_assessments"][0]["validation_evidence"] = [
        {
            "command": "pytest tests/test_client.py",
            "result": "failed",
            "explanation": "The focused public behavior failed.",
        }
    ]
    with pytest.raises(ContractValidationError, match="non-passing validation"):
        validate_assessment(validator_v3, failed)


def test_v3_fix_request_routes_must_gap_and_builds_context(validator_v3):
    payload = assessment_v3()
    payload["obligation_assessments"][0] = assessment_item(
        "O01", implementation_status="missing"
    )
    payload["repair_requests"] = [
        {
            "id": "Q01",
            "finding_refs": ["O01"],
            "action": "fix",
            "problem": "The timeout is ignored.",
            "required_outcome": "Apply the timeout.",
            "preserve": ["Calls without a timeout remain valid."],
            "validation": ["Exercise construction and one retry."],
        }
    ]
    payload["verdict"] = "request_changes"
    payload["verdict_rationale"] = "A localized must gap has an executable repair."
    validate_assessment(validator_v3, payload)

    contexts = validator_v3.build_request_contexts(
        ledger=ledger_v3(), assessment=payload
    )
    assert contexts[0]["order"] == 1
    assert [item["id"] for item in contexts[0]["obligations"]] == ["O01"]
    assert contexts[0]["source_ids"] == ["S01"]


def test_v3_should_gap_cannot_trigger_repair(validator_v3):
    payload = assessment_v3()
    payload["obligation_assessments"][1] = assessment_item(
        "O02", implementation_status="missing"
    )
    payload["repair_requests"] = [
        {
            "id": "Q01",
            "finding_refs": ["O02"],
            "action": "fix",
            "problem": "The optional quality expectation is absent.",
            "required_outcome": "Implement it.",
            "preserve": [],
            "validation": ["Run a focused check."],
        }
    ]
    payload["verdict"] = "request_changes"
    with pytest.raises(ContractValidationError, match="uses should obligation"):
        validate_assessment(validator_v3, payload)


def test_v3_unclear_must_routes_to_abstain(validator_v3):
    payload = assessment_v3()
    payload["obligation_assessments"][0] = assessment_item(
        "O01", source_status="unclear"
    )
    payload["verdict"] = "abstain"
    payload["verdict_rationale"] = "Public sources permit incompatible timeout meanings."
    validate_assessment(validator_v3, payload)


def test_v3_effective_set_records_omissions_and_source_dispositions(validator_v3):
    payload = assessment_v3()
    payload["obligation_assessments"][1] = assessment_item(
        "O02", source_status="unsupported"
    )
    effective = validator_v3.build_effective_review_set(
        task_id="task-a",
        ledger_sha256=LEDGER_SHA,
        assessment_sha256="4" * 64,
        ledger=ledger_v3(),
        assessment=payload,
    )
    assert effective["effective_obligations"] == ["O01"]
    assert effective["nonbinding_obligations"] == ["O02"]
    assert effective["unresolved_obligations"] == []
    assert effective["late_obligation_count"] == 0


def test_v3_repair_audit_closes_fix_request_and_exact_diff(validator_v3):
    assessment = assessment_v3()
    assessment["obligation_assessments"][0] = assessment_item(
        "O01", implementation_status="missing"
    )
    assessment["repair_requests"] = [
        {
            "id": "Q01",
            "finding_refs": ["O01"],
            "action": "fix",
            "problem": "The timeout is ignored.",
            "required_outcome": "Apply the timeout.",
            "preserve": ["Omitted timeouts retain the default."],
            "validation": ["Exercise explicit and omitted timeouts."],
        }
    ]
    assessment["verdict"] = "request_changes"
    validate_assessment(validator_v3, assessment)
    effective = validator_v3.build_effective_review_set(
        task_id="task-a",
        ledger_sha256=LEDGER_SHA,
        assessment_sha256="4" * 64,
        ledger=ledger_v3(),
        assessment=assessment,
    )
    audit = {
        "schema_version": "3.0",
        "task_id": "task-a",
        "candidate_patch_sha256": PATCH_SHA,
        "assessment_sha256": "4" * 64,
        "effective_review_set_sha256": "5" * 64,
        "changed_paths": ["src/client.py"],
        "changes": [
            {
                "change_id": "C01",
                "request_refs": ["Q01"],
                "finding_refs": ["O01"],
                "decision": "implement",
                "before_behavior": "The explicit timeout is ignored.",
                "after_behavior": "The explicit timeout is applied.",
                "preserve": ["Omitted timeouts retain the default."],
                "out_of_request_scope": False,
                "scope_rationale": None,
                "changed_paths": ["src/client.py"],
                "validation": [
                    {
                        "command": "pytest -k timeout",
                        "result": "passed",
                        "expected": "Both timeout forms work.",
                        "actual": "Both timeout forms passed.",
                        "log_path": None,
                    }
                ],
            }
        ],
        "unaddressed_requests": [],
        "validation_summary": "The focused timeout validation passed.",
    }
    validator_v3.validate_repair_audit(
        audit,
        task_id="task-a",
        p0_sha256=PATCH_SHA,
        assessment_sha256="4" * 64,
        effective_review_set_sha256="5" * 64,
        ledger=ledger_v3(),
        assessment=assessment,
        effective_review_set=effective,
        actual_changed_paths={"src/client.py"},
    )

    invalid = copy.deepcopy(audit)
    invalid["changed_paths"] = []
    with pytest.raises(ContractValidationError, match="changed_paths do not match"):
        validator_v3.validate_repair_audit(
            invalid,
            task_id="task-a",
            p0_sha256=PATCH_SHA,
            assessment_sha256="4" * 64,
            effective_review_set_sha256="5" * 64,
            ledger=ledger_v3(),
            assessment=assessment,
            effective_review_set=effective,
            actual_changed_paths={"src/client.py"},
        )


def test_v3_no_change_is_limited_to_investigate_requests(validator_v3):
    assessment = assessment_v3()
    assessment["obligation_assessments"][0] = assessment_item(
        "O01", implementation_status="uncertain"
    )
    assessment["repair_requests"] = [
        {
            "id": "Q01",
            "finding_refs": ["O01"],
            "action": "investigate",
            "problem": "Timeout propagation cannot be established statically.",
            "required_outcome": "Determine whether retries apply it.",
            "preserve": [],
            "validation": ["Run a focused retry probe."],
        }
    ]
    assessment["verdict"] = "request_changes"
    validate_assessment(validator_v3, assessment)
    effective = validator_v3.build_effective_review_set(
        task_id="task-a",
        ledger_sha256=LEDGER_SHA,
        assessment_sha256="4" * 64,
        ledger=ledger_v3(),
        assessment=assessment,
    )
    audit = {
        "schema_version": "3.0",
        "task_id": "task-a",
        "candidate_patch_sha256": PATCH_SHA,
        "assessment_sha256": "4" * 64,
        "effective_review_set_sha256": "5" * 64,
        "changed_paths": [],
        "changes": [
            {
                "change_id": "C01",
                "request_refs": ["Q01"],
                "finding_refs": ["O01"],
                "decision": "no_change",
                "before_behavior": "Retry propagation was uncertain.",
                "after_behavior": "A public probe confirmed propagation.",
                "preserve": [],
                "out_of_request_scope": False,
                "scope_rationale": None,
                "changed_paths": [],
                "validation": [
                    {
                        "command": "pytest -k retry_timeout",
                        "result": "passed",
                        "expected": "Retries apply the timeout.",
                        "actual": "The timeout reached the retry call.",
                        "log_path": None,
                    }
                ],
            }
        ],
        "unaddressed_requests": [],
        "validation_summary": "No source-grounded defect was found.",
    }
    validator_v3.validate_repair_audit(
        audit,
        task_id="task-a",
        p0_sha256=PATCH_SHA,
        assessment_sha256="4" * 64,
        effective_review_set_sha256="5" * 64,
        ledger=ledger_v3(),
        assessment=assessment,
        effective_review_set=effective,
        actual_changed_paths=set(),
    )

    fix_assessment = copy.deepcopy(assessment)
    fix_assessment["repair_requests"][0]["action"] = "fix"
    with pytest.raises(ContractValidationError, match="only for investigate"):
        validator_v3.validate_repair_audit(
            audit,
            task_id="task-a",
            p0_sha256=PATCH_SHA,
            assessment_sha256="4" * 64,
            effective_review_set_sha256="5" * 64,
            ledger=ledger_v3(),
            assessment=fix_assessment,
            effective_review_set=effective,
            actual_changed_paths=set(),
        )
