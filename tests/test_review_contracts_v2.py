import copy

import pytest

from external_agents_eval.workflows.solve_review_repair.config import (
    DEFAULT_PROMPT_DIR,
    DEFAULT_SCHEMA_DIR,
)
from external_agents_eval.workflows.solve_review_repair.contracts_v2 import (
    changed_paths_between_patches,
)
from external_agents_eval.workflows.solve_review_repair.validators import (
    ContractValidationError,
    ContractValidator,
)

POLICY_SHA = "1" * 64
P0_SHA = "2" * 64
LEDGER_SHA = "3" * 64


def ledger_v2():
    return {
        "schema_version": "2.0",
        "task_id": "task-a",
        "source_policy_sha256": POLICY_SHA,
        "feature_summary": "Expose and apply a timeout.",
        "scope": {"in_scope": ["client"], "out_of_scope": [], "assumptions": []},
        "sources": [
            {
                "id": "S01",
                "kind": "issue",
                "locator": {"field": "body", "quote": "Accept a timeout."},
            }
        ],
        "obligations": [
            {
                "id": "O01",
                "statement": "The public client accepts a timeout.",
                "category": "public_api",
                "acceptance_criteria": [
                    {
                        "id": "O01.C01",
                        "statement": "Construction accepts a timeout.",
                        "criticality": "must",
                        "basis_kind": "explicit_issue",
                        "source_refs": ["S01"],
                        "derivation": "The issue explicitly requires the input.",
                        "open_questions": [],
                        "confidence": "high",
                    }
                ],
                "affected_surfaces": ["Client"],
                "dependencies": [],
            }
        ],
    }


def evidence():
    return [
        {
            "path": "src/client.py",
            "symbol": "Client",
            "line_start": 1,
            "line_end": 2,
            "explanation": "The constructor stores the value.",
        }
    ]


def source_check(status="supported"):
    return {
        "status": status,
        "source_refs": ["S01"],
        "rationale": "The issue directly establishes this behavior.",
        "compliant_but_rejected_example": None,
        "satisfies_but_violates_example": None,
    }


def assessment_v2(*, status="covered", verdict="approve"):
    item = {
        "criterion_ref": "O01.C01",
        "source_check": source_check(),
        "implementation_status": status,
        "implementation_evidence": evidence(),
        "test_evidence": [],
        "validation_note": "Inspected the public constructor." if status == "covered" else None,
        "gap": None if status == "covered" else "The constructor ignores the timeout.",
        "confidence": "high",
    }
    payload = {
        "schema_version": "2.0",
        "task_id": "task-a",
        "patch_sha256": P0_SHA,
        "ledger_sha256": LEDGER_SHA,
        "additional_sources": [],
        "criterion_assessments": [item],
        "ledger_challenges": [],
        "unexpected_regressions": [],
        "late_obligation_candidates": [],
        "verdict": verdict,
        "verdict_rationale": "The effective requirement is covered.",
        "abstain_reason": None,
        "repair_requests": [],
    }
    if verdict in {"request_changes", "replan"}:
        payload["repair_requests"] = [
            {
                "id": "Q01",
                "priority": 1,
                "finding_refs": ["O01.C01"],
                "criterion_refs": ["O01.C01"],
                "source_refs": ["S01"],
                "action_kind": "fix" if status == "missing" else "investigate",
                "problem": "The timeout is not propagated.",
                "requested_outcome": "Apply it to construction and retries.",
                "preserve_behaviors": ["Existing default calls remain valid."],
                "suggested_validation": ["Exercise construction and one retry."],
            }
        ]
    return payload


@pytest.fixture
def validator_v2():
    return ContractValidator(DEFAULT_SCHEMA_DIR, contract_version="2.0")


def test_v2_schema_bundle_is_independent_from_v1(validator_v2):
    assert set(validator_v2.SCHEMAS) == {
        "source_policy",
        "build_input",
        "ledger",
        "assessment_input",
        "assessment",
        "effective_review_set",
        "repair_input",
        "repair_audit",
    }
    assert ContractValidator(DEFAULT_SCHEMA_DIR).schema("ledger")["title"] == ("ReviewObligations")


def test_v2_stage_a_prompt_exposes_identifier_and_category_contracts():
    prompt = (DEFAULT_PROMPT_DIR / "build_obligations_v2.md").read_text(encoding="utf-8")

    assert "IDENTIFIERS, LIMITS, AND ENUMS (HARD REQUIREMENTS)" in prompt
    assert "Every sources[].id" in prompt
    assert "Never use `Ixx`, `Rxx`" in prompt
    assert "Every obligations[].id" in prompt
    assert "acceptance criterion ID" in prompt
    assert "Copy `task_id` and `source_policy_sha256` byte-for-byte" in prompt
    assert "complete dependency graph must be acyclic" in prompt
    assert "basis_kind=`explicit_issue` must reference at least one issue" in prompt
    assert "Use criticality=must for an explicit requested behavior" in prompt
    assert "Few-shots are illustrative and are not a substitute for the schema" in prompt
    assert "{{REVIEW_OBLIGATIONS_OUTPUT_SCHEMA}}" in prompt
    for category in (
        "public_api",
        "configuration",
        "data_flow",
        "core_behavior",
        "error_handling",
        "backward_compatibility",
        "integration",
        "serialization",
        "concurrency",
        "performance",
        "documentation",
    ):
        assert f"`{category}`" in prompt


def test_v2_stage_b_prompt_exposes_cross_field_contracts_and_scratch_policy():
    prompt = (DEFAULT_PROMPT_DIR / "assess_patch_v2.md").read_text(encoding="utf-8")

    assert "Do not modify or delete any base-tracked file or any file belonging to frozen P0" in prompt
    assert "disposable untracked cache, scratch, or test-output files" in prompt
    assert "ASSESSMENT CONTRACT (HARD REQUIREMENTS)" in prompt
    assert "Copy `task_id`, `patch_sha256`, and `ledger_sha256` byte-for-byte" in prompt
    assert "challenge `D[0-9]{2}`" in prompt
    assert "unexpected regression\n  `R[0-9]{2}`" in prompt
    assert "late criterion `L[0-9]{2}.C[0-9]{2}`" in prompt
    assert "repair request `Q[0-9]{2}`" in prompt
    assert "Every non-supported status\n  requires exactly one challenge" in prompt
    assert "criterion_refs must exactly equal all non-regression entries" in prompt
    assert "request priorities must be exactly the continuous sequence 1..N" in prompt
    assert "Apply verdict routing exactly" in prompt
    assert "The effective criterion set is exact" in prompt
    assert "Never let P0 quality or test results change the source judgment" in prompt
    assert "a missing new feature is not a regression" in prompt
    assert "Never omit either field" in prompt
    assert "covered late criterion" in prompt
    assert "Regression severity describes impact; blocking independently states" in prompt
    assert "`replan` only for an evidenced failure of the core design direction" in prompt
    assert "additional_sources<=32" in prompt
    assert '`implementation_evidence` item =' in prompt
    assert '`test_evidence` item =' in prompt
    assert '"symbol": string|null' in prompt
    assert '"test_name": non-empty string' in prompt
    assert "{{PATCH_ASSESSMENT_OUTPUT_SCHEMA}}" in prompt


def test_v2_covered_requires_passing_test_evidence(validator_v2):
    payload = assessment_v2()
    payload["criterion_assessments"][0]["validation_note"] = None
    payload["criterion_assessments"][0]["test_evidence"] = [
        {
            "path": "tests/test_client.py",
            "test_name": "test_timeout",
            "command": "python -m pytest tests/test_client.py::test_timeout -q",
            "result": "failed",
            "explanation": "The focused assertion failed.",
        }
    ]

    with pytest.raises(ContractValidationError, match="non-passing tests"):
        validator_v2.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )


def test_v2_blocker_regression_must_be_blocking(validator_v2):
    payload = assessment_v2()
    payload["unexpected_regressions"] = [
        {
            "id": "R01",
            "problem": "A public call now raises.",
            "severity": "blocker",
            "blocking": False,
            "source_refs": ["S01"],
            "implementation_evidence": evidence(),
        }
    ]

    with pytest.raises(ContractValidationError, match="True was expected"):
        validator_v2.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )


def test_v2_late_criterion_cannot_be_not_applicable(validator_v2):
    payload = assessment_v2()
    payload["late_obligation_candidates"] = [
        {
            "id": "L01.C01",
            "statement": "Keep the public constructor importable.",
            "criticality": "should",
            "basis_kind": "explicit_issue",
            "source_refs": ["S01"],
            "derivation": "The public issue establishes the boundary.",
            "source_check": source_check(),
            "implementation_status": "not_applicable",
            "implementation_evidence": [],
            "test_evidence": [],
            "gap": None,
        }
    ]

    with pytest.raises(ContractValidationError, match="not one of"):
        validator_v2.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )


def test_v2_repair_prompt_exposes_audit_semantic_invariants():
    prompt = (DEFAULT_PROMPT_DIR / "repair_once_v2.md").read_text(encoding="utf-8")

    assert "AUDIT SEMANTIC INVARIANTS (HARD REQUIREMENTS)" in prompt
    assert "change_id values must be unique and match `C[0-9]{2}`" in prompt
    assert "decision=`implement` permits only" in prompt
    assert "Top-level changed_paths must equal the exact P0-to-P1" in prompt
    assert "never both and never neither" in prompt
    assert "source/reference closure" in prompt


def test_v2_ledger_binds_each_atomic_criterion_to_sources(validator_v2):
    payload = ledger_v2()
    validator_v2.validate_ledger(payload, task_id="task-a", source_policy_sha256=POLICY_SHA)

    payload["obligations"][0]["acceptance_criteria"][0]["source_refs"] = ["S99"]
    with pytest.raises(ContractValidationError, match="unknown sources"):
        validator_v2.validate_ledger(payload, task_id="task-a", source_policy_sha256=POLICY_SHA)


def test_v2_unspecified_basis_cannot_be_a_must(validator_v2):
    payload = ledger_v2()
    criterion = payload["obligations"][0]["acceptance_criteria"][0]
    criterion["basis_kind"] = "unspecified"
    with pytest.raises(ContractValidationError, match="both be unspecified"):
        validator_v2.validate_ledger(payload, task_id="task-a", source_policy_sha256=POLICY_SHA)


def test_issue_source_anchor_requires_an_exact_quote(validator_v2):
    validator_v2.validate_issue_source_anchors(
        ledger_v2()["sources"], title="Timeout", body="Accept a timeout."
    )
    with pytest.raises(ContractValidationError, match="not exact"):
        validator_v2.validate_issue_source_anchors(
            ledger_v2()["sources"], title="Timeout", body="Different requirement."
        )


def test_repository_source_path_cannot_escape_the_repository(validator_v2):
    payload = ledger_v2()
    payload["sources"] = [
        {
            "id": "S01",
            "kind": "repository",
            "locator": {
                "path": "../hidden/answer.py",
                "base_revision": "a" * 40,
                "line_start": 1,
                "line_end": 1,
                "quote": "secret",
                "file_sha256": "0" * 64,
            },
        }
    ]
    criterion = payload["obligations"][0]["acceptance_criteria"][0]
    criterion["basis_kind"] = "repository_contract"

    with pytest.raises(ContractValidationError, match="not valid"):
        validator_v2.validate_ledger(
            payload,
            task_id="task-a",
            source_policy_sha256=POLICY_SHA,
        )


def test_overstated_detail_can_be_nonbinding_but_remains_traceable(validator_v2):
    payload = assessment_v2()
    item = payload["criterion_assessments"][0]
    item["source_check"] = source_check("overstated")
    item["implementation_status"] = "not_applicable"
    item["implementation_evidence"] = []
    item["validation_note"] = None
    payload["ledger_challenges"] = [
        {
            "id": "D01",
            "criterion_ref": "O01.C01",
            "kind": "unsupported_specificity",
            "source_refs": ["S01"],
            "explanation": "The quote does not establish the added detail.",
            "alternative_interpretations": ["The existing public default may remain."],
            "disposition": "nonbinding",
            "replacement_refs": [],
        }
    ]

    validator_v2.validate_assessment(
        payload,
        task_id="task-a",
        patch_sha256=P0_SHA,
        ledger_sha256=LEDGER_SHA,
        ledger=ledger_v2(),
    )
    effective = validator_v2.build_effective_review_set(
        task_id="task-a",
        ledger_sha256=LEDGER_SHA,
        assessment_sha256="4" * 64,
        ledger=ledger_v2(),
        assessment=payload,
    )
    assert effective["effective_criteria"] == []
    assert effective["nonbinding_criteria"] == ["O01.C01"]


def test_source_check_cannot_silently_replace_the_frozen_criterion_source(validator_v2):
    payload = assessment_v2()
    payload["additional_sources"] = [
        {
            "id": "S02",
            "kind": "issue",
            "locator": {"field": "body", "quote": "Accept a timeout."},
        }
    ]
    payload["criterion_assessments"][0]["source_check"]["source_refs"] = ["S02"]

    with pytest.raises(ContractValidationError, match="omits frozen criterion sources: S01"):
        validator_v2.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )


def test_weakened_requirement_requires_a_source_grounded_replacement(validator_v2):
    payload = assessment_v2(status="missing", verdict="request_changes")
    item = payload["criterion_assessments"][0]
    item["source_check"] = source_check("understated")
    item["implementation_status"] = "not_applicable"
    item["implementation_evidence"] = []
    item["gap"] = None
    payload["ledger_challenges"] = [
        {
            "id": "D01",
            "criterion_ref": "O01.C01",
            "kind": "weakened_requirement",
            "source_refs": ["S01"],
            "explanation": "The criterion covers only construction, not retries.",
            "alternative_interpretations": [],
            "disposition": "replaced",
            "replacement_refs": ["L01.C01"],
        }
    ]
    payload["late_obligation_candidates"] = [
        {
            "id": "L01.C01",
            "statement": "Construction and retries apply the timeout.",
            "criticality": "must",
            "basis_kind": "explicit_issue",
            "source_refs": ["S01"],
            "derivation": "The complete issue requires both paths.",
            "source_check": source_check(),
            "implementation_status": "missing",
            "implementation_evidence": evidence(),
            "test_evidence": [],
            "gap": "Retries ignore the timeout.",
        }
    ]
    request = payload["repair_requests"][0]
    request["finding_refs"] = ["L01.C01"]
    request["criterion_refs"] = ["L01.C01"]

    validator_v2.validate_assessment(
        payload,
        task_id="task-a",
        patch_sha256=P0_SHA,
        ledger_sha256=LEDGER_SHA,
        ledger=ledger_v2(),
    )
    effective = validator_v2.build_effective_review_set(
        task_id="task-a",
        ledger_sha256=LEDGER_SHA,
        assessment_sha256="4" * 64,
        ledger=ledger_v2(),
        assessment=payload,
    )
    assert [item["criterion_ref"] for item in effective["effective_criteria"]] == ["L01.C01"]

    wrong_basis = copy.deepcopy(payload)
    wrong_basis["late_obligation_candidates"][0]["basis_kind"] = "repository_contract"
    with pytest.raises(ContractValidationError, match="repository basis requires"):
        validator_v2.validate_assessment(
            wrong_basis,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )


def test_uncertain_implementation_uses_investigate_or_abstain(validator_v2):
    payload = assessment_v2(status="uncertain", verdict="request_changes")
    validator_v2.validate_assessment(
        payload,
        task_id="task-a",
        patch_sha256=P0_SHA,
        ledger_sha256=LEDGER_SHA,
        ledger=ledger_v2(),
    )
    payload["repair_requests"][0]["action_kind"] = "fix"
    with pytest.raises(ContractValidationError, match="must investigate"):
        validator_v2.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )

    abstain = assessment_v2(status="uncertain", verdict="approve")
    abstain.update(
        {
            "verdict": "abstain",
            "verdict_rationale": "The behavior cannot be verified with public evidence.",
            "abstain_reason": "The only applicable runtime is unavailable.",
        }
    )
    validator_v2.validate_assessment(
        abstain,
        task_id="task-a",
        patch_sha256=P0_SHA,
        ledger_sha256=LEDGER_SHA,
        ledger=ledger_v2(),
    )


def test_investigate_request_cannot_mix_an_evidenced_regression(validator_v2):
    payload = assessment_v2(status="uncertain", verdict="request_changes")
    payload["unexpected_regressions"] = [
        {
            "id": "R01",
            "problem": "P0 breaks the existing default construction path.",
            "severity": "high",
            "blocking": True,
            "source_refs": ["S01"],
            "implementation_evidence": evidence(),
        }
    ]
    payload["repair_requests"][0]["finding_refs"].append("R01")

    with pytest.raises(
        ContractValidationError,
        match="cannot investigate an already evidenced unexpected regression",
    ):
        validator_v2.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )


def test_repair_request_requires_a_discriminating_validation(validator_v2):
    payload = assessment_v2(status="missing", verdict="request_changes")
    payload["repair_requests"][0]["suggested_validation"] = []

    with pytest.raises(ContractValidationError, match="should be non-empty"):
        validator_v2.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )


def test_repair_request_cannot_mismatch_findings_criteria_sources_or_action(validator_v2):
    payload = assessment_v2(status="missing", verdict="request_changes")
    payload["additional_sources"] = [
        {
            "id": "S02",
            "kind": "issue",
            "locator": {"field": "body", "quote": "Accept a timeout."},
        }
    ]

    mismatched_refs = copy.deepcopy(payload)
    mismatched_refs["repair_requests"][0]["criterion_refs"] = []
    with pytest.raises(ContractValidationError, match="must exactly match"):
        validator_v2.validate_assessment(
            mismatched_refs,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )

    unrelated_source = copy.deepcopy(payload)
    unrelated_source["repair_requests"][0]["source_refs"] = ["S02"]
    with pytest.raises(ContractValidationError, match="omits finding sources: S01"):
        validator_v2.validate_assessment(
            unrelated_source,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )

    wrong_action = copy.deepcopy(payload)
    wrong_action["criterion_assessments"][0]["implementation_status"] = "covered"
    wrong_action["criterion_assessments"][0]["gap"] = None
    wrong_action["criterion_assessments"][0]["validation_note"] = "Inspected directly."
    with pytest.raises(ContractValidationError, match="cannot fix criteria"):
        validator_v2.validate_assessment(
            wrong_action,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )


def test_repair_audit_binds_sources_requests_and_actual_diff(validator_v2):
    assessment = assessment_v2(status="missing", verdict="request_changes")
    validator_v2.validate_assessment(
        assessment,
        task_id="task-a",
        patch_sha256=P0_SHA,
        ledger_sha256=LEDGER_SHA,
        ledger=ledger_v2(),
    )
    effective = validator_v2.build_effective_review_set(
        task_id="task-a",
        ledger_sha256=LEDGER_SHA,
        assessment_sha256="4" * 64,
        ledger=ledger_v2(),
        assessment=assessment,
    )
    audit = {
        "schema_version": "2.0",
        "task_id": "task-a",
        "p0_sha256": P0_SHA,
        "assessment_sha256": "4" * 64,
        "effective_review_set_sha256": "5" * 64,
        "additional_sources": [],
        "changed_paths": ["src/client.py"],
        "changes": [
            {
                "change_id": "C01",
                "request_refs": ["Q01"],
                "criterion_refs": ["O01.C01"],
                "risk_tags": ["public_api"],
                "before_behavior": "The timeout is ignored.",
                "after_behavior": "The timeout is applied.",
                "source_refs": ["S01"],
                "source_check": "supported",
                "counterevidence": [],
                "preserve_behaviors": ["Calls without a timeout remain valid."],
                "decision": "implement",
                "changed_paths": ["src/client.py"],
                "validation": [
                    {
                        "command": "pytest tests/test_client.py",
                        "exit_code": 0,
                        "expected": "The retry receives the timeout.",
                        "actual": "The retry received the timeout.",
                        "log_path": None,
                    }
                ],
            }
        ],
        "unaddressed_requests": [],
        "validation_summary": "The focused public regression test passed.",
    }
    validator_v2.validate_repair_audit(
        audit,
        task_id="task-a",
        p0_sha256=P0_SHA,
        assessment_sha256="4" * 64,
        effective_review_set_sha256="5" * 64,
        ledger=ledger_v2(),
        assessment=assessment,
        effective_review_set=effective,
        actual_changed_paths={"src/client.py"},
    )

    broken = copy.deepcopy(audit)
    broken["changes"][0]["request_refs"] = []
    broken["changes"][0]["risk_tags"].append("out_of_request_scope")
    with pytest.raises(ContractValidationError, match="omits requests"):
        validator_v2.validate_repair_audit(
            broken,
            task_id="task-a",
            p0_sha256=P0_SHA,
            assessment_sha256="4" * 64,
            effective_review_set_sha256="5" * 64,
            ledger=ledger_v2(),
            assessment=assessment,
            effective_review_set=effective,
            actual_changed_paths={"src/client.py"},
        )

    unaudited_path = copy.deepcopy(audit)
    unaudited_path["changed_paths"].append("src/helper.py")
    with pytest.raises(ContractValidationError, match="without an implemented change record"):
        validator_v2.validate_repair_audit(
            unaudited_path,
            task_id="task-a",
            p0_sha256=P0_SHA,
            assessment_sha256="4" * 64,
            effective_review_set_sha256="5" * 64,
            ledger=ledger_v2(),
            assessment=assessment,
            effective_review_set=effective,
            actual_changed_paths={"src/client.py", "src/helper.py"},
        )

    no_validation = copy.deepcopy(audit)
    no_validation["changes"][0]["validation"] = []
    with pytest.raises(ContractValidationError, match="has no validation record"):
        validator_v2.validate_repair_audit(
            no_validation,
            task_id="task-a",
            p0_sha256=P0_SHA,
            assessment_sha256="4" * 64,
            effective_review_set_sha256="5" * 64,
            ledger=ledger_v2(),
            assessment=assessment,
            effective_review_set=effective,
            actual_changed_paths={"src/client.py"},
        )

    mismatched_criterion = copy.deepcopy(audit)
    mismatched_criterion["changes"][0]["criterion_refs"] = []
    with pytest.raises(ContractValidationError, match="omits its requested criteria"):
        validator_v2.validate_repair_audit(
            mismatched_criterion,
            task_id="task-a",
            p0_sha256=P0_SHA,
            assessment_sha256="4" * 64,
            effective_review_set_sha256="5" * 64,
            ledger=ledger_v2(),
            assessment=assessment,
            effective_review_set=effective,
            actual_changed_paths={"src/client.py"},
        )

    unrelated_criterion = copy.deepcopy(audit)
    unrelated_criterion["additional_sources"] = [
        {
            "id": "S02",
            "kind": "issue",
            "locator": {"field": "body", "quote": "Accept a timeout."},
        }
    ]
    unrelated_criterion["changes"][0]["criterion_refs"] = []
    unrelated_criterion["changes"][0]["risk_tags"].append("out_of_request_scope")
    unrelated_criterion["changes"][0]["source_refs"] = ["S02"]
    with pytest.raises(ContractValidationError, match="omits request/criterion sources"):
        validator_v2.validate_repair_audit(
            unrelated_criterion,
            task_id="task-a",
            p0_sha256=P0_SHA,
            assessment_sha256="4" * 64,
            effective_review_set_sha256="5" * 64,
            ledger=ledger_v2(),
            assessment=assessment,
            effective_review_set=effective,
            actual_changed_paths={"src/client.py"},
        )


def test_duplicate_unexpected_regression_ids_are_rejected(validator_v2):
    payload = assessment_v2(status="missing", verdict="request_changes")
    regression = {
        "id": "R01",
        "problem": "An existing public call now raises.",
        "severity": "high",
        "blocking": True,
        "source_refs": ["S01"],
        "implementation_evidence": evidence(),
    }
    payload["unexpected_regressions"] = [regression, copy.deepcopy(regression)]

    with pytest.raises(ContractValidationError, match="unexpected regression IDs"):
        validator_v2.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256=P0_SHA,
            ledger_sha256=LEDGER_SHA,
            ledger=ledger_v2(),
        )


def test_changed_paths_between_patches_ignores_unchanged_p0_files():
    common = b"diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    after = common + b"diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert changed_paths_between_patches(common, after) == {"b.py"}
