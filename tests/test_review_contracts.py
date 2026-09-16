import copy
import json
from pathlib import Path

import pytest

from external_agents_eval.workflows.solve_review_repair.config import (
    DEFAULT_PROMPT_DIR,
    DEFAULT_SCHEMA_DIR,
)
from external_agents_eval.workflows.solve_review_repair.prompt_renderer import PromptRenderer
from external_agents_eval.workflows.solve_review_repair.validators import (
    ContractValidationError,
    ContractValidator,
    extract_single_json_object,
)


def ledger():
    return {
        "schema_version": "1.0",
        "task_id": "task-a",
        "feature_summary": "Expose and apply a timeout.",
        "scope": {"in_scope": ["client"], "out_of_scope": [], "assumptions": []},
        "obligations": [
            {
                "id": "O01",
                "statement": "The public client accepts a timeout.",
                "origins": [
                    {
                        "type": "explicit_issue",
                        "evidence": "accept a timeout",
                        "rationale": "The issue requires the input.",
                    }
                ],
                "category": "public_api",
                "criticality": "must",
                "acceptance_criteria": ["Construction accepts a timeout."],
                "affected_surfaces": ["Client"],
                "dependencies": [],
                "confidence": "high",
            },
            {
                "id": "O02",
                "statement": "Retries apply the timeout.",
                "origins": [
                    {
                        "type": "explicit_issue",
                        "evidence": "apply it to retries",
                        "rationale": "The issue names retries.",
                    }
                ],
                "category": "data_flow",
                "criticality": "must",
                "acceptance_criteria": ["Every retry uses the timeout."],
                "affected_surfaces": ["retry"],
                "dependencies": ["O01"],
                "confidence": "high",
            },
        ],
    }


def assessment():
    implementation = [
        {
            "path": "src/client.py",
            "symbol": "Client",
            "line_start": 1,
            "line_end": 2,
            "explanation": "Stores and uses the value.",
        }
    ]
    validation = "The public constructor and retry call were inspected."
    return {
        "schema_version": "1.0",
        "task_id": "task-a",
        "patch_sha256": "a" * 64,
        "ledger_sha256": "b" * 64,
        "obligation_assessments": [
            {
                "obligation_id": "O01",
                "status": "covered",
                "implementation_evidence": implementation,
                "test_evidence": [],
                "validation_note": validation,
                "gap": None,
                "confidence": "high",
            },
            {
                "obligation_id": "O02",
                "status": "covered",
                "implementation_evidence": implementation,
                "test_evidence": [],
                "validation_note": validation,
                "gap": None,
                "confidence": "high",
            },
        ],
        "unexpected_regressions": [],
        "late_obligation_candidates": [],
        "verdict": "approve",
        "verdict_rationale": "Every must obligation is covered.",
        "repair_requests": [],
    }


@pytest.fixture
def validator():
    return ContractValidator(DEFAULT_SCHEMA_DIR)


def test_all_bundled_schemas_self_validate(validator):
    assert set(validator.SCHEMAS) == {
        "build_input",
        "ledger",
        "assessment_input",
        "assessment",
        "repair_input",
    }


def test_final_output_must_be_exactly_one_json_object(validator):
    with pytest.raises(ContractValidationError, match="JSON parse error"):
        validator.parse('commentary\n{"schema_version":"1.0"}')
    with pytest.raises(ContractValidationError, match="one JSON object"):
        validator.parse("[]")


def test_exact_json_object_does_not_need_normalization():
    raw = '{"schema_version":"1.0"}'
    assert extract_single_json_object(raw) == (raw, None)


def test_single_fenced_json_object_is_unwrapped_from_commentary():
    raw = 'Here is the result.\n```json\n{"schema_version":"1.0"}\n```'
    assert extract_single_json_object(raw) == (
        '{"schema_version":"1.0"}',
        "fenced_json",
    )


def test_spaced_json_fence_is_supported():
    raw = '``` json\n{"schema_version":"1.0"}\n```'
    assert extract_single_json_object(raw) == (
        '{"schema_version":"1.0"}',
        "fenced_json",
    )


def test_single_embedded_json_object_is_decoded_after_commentary():
    raw = 'Emitting the corrected object now:\n{"outer":{"inner":"}"}}\nDone.'
    assert extract_single_json_object(raw) == (
        '{"outer":{"inner":"}"}}',
        "embedded_json_object",
    )


def test_multiple_embedded_json_objects_are_not_guessed():
    raw = 'first {"value":1}\nsecond {"value":2}'
    assert extract_single_json_object(raw) == (raw, None)


def test_parseable_object_inside_malformed_outer_braces_is_not_guessed():
    raw = 'broken {wrapper: {"value":1}}'
    assert extract_single_json_object(raw) == (raw, None)


def test_multiple_valid_fenced_objects_are_not_guessed():
    raw = '```json\n{"first":true}\n```\n```json\n{"second":true}\n```'
    assert extract_single_json_object(raw) == (raw, None)


def test_fenced_array_is_not_unwrapped():
    raw = "commentary\n```json\n[]\n```"
    assert extract_single_json_object(raw) == (raw, None)


@pytest.mark.parametrize(
    "prompt_name",
    ["build_obligations_v1.md", "assess_patch_v1.md", "correct_json_v1.md"],
)
def test_structured_role_prompts_require_one_json_fence(prompt_name):
    prompt = (DEFAULT_PROMPT_DIR / prompt_name).read_text(encoding="utf-8")
    assert "inside exactly one JSON Markdown code block" in prompt
    assert "```json" in prompt
    assert "Do not add" in prompt


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["obligations"].append(copy.deepcopy(value["obligations"][0])),
            "unique",
        ),
        (lambda value: value["obligations"][1]["dependencies"].append("O99"), "dangling"),
        (lambda value: value["obligations"][0]["dependencies"].append("O02"), "cycle"),
    ],
)
def test_ledger_semantics_reject_duplicate_dangling_and_cycle(validator, mutate, message):
    payload = ledger()
    mutate(payload)
    with pytest.raises(ContractValidationError, match=message):
        validator.validate_ledger(payload, task_id="task-a")


def test_schema_rejects_additional_fields(validator):
    payload = ledger()
    payload["unexpected"] = True
    with pytest.raises(ContractValidationError, match="Additional properties"):
        validator.validate_ledger(payload, task_id="task-a")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["obligation_assessments"].pop(), "missing obligation"),
        (
            lambda value: value["obligation_assessments"].append(
                copy.deepcopy(value["obligation_assessments"][0])
            ),
            "duplicates",
        ),
        (
            lambda value: value["obligation_assessments"][0].update(
                {"implementation_evidence": []}
            ),
            "without implementation evidence",
        ),
        (lambda value: value.update({"patch_sha256": "c" * 64}), "patch_sha256"),
        (lambda value: value.update({"ledger_sha256": "c" * 64}), "ledger_sha256"),
    ],
)
def test_assessment_cross_field_contracts(validator, mutate, message):
    payload = assessment()
    mutate(payload)
    with pytest.raises(ContractValidationError, match=message):
        validator.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256="a" * 64,
            ledger_sha256="b" * 64,
            ledger=ledger(),
        )


def test_approve_rejects_must_gap(validator):
    payload = assessment()
    payload["obligation_assessments"][1].update(
        {
            "status": "missing",
            "implementation_evidence": [],
            "validation_note": None,
            "gap": "Retries ignore the value.",
        }
    )
    with pytest.raises(ContractValidationError, match="approve conflicts"):
        validator.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256="a" * 64,
            ledger_sha256="b" * 64,
            ledger=ledger(),
        )


def test_repair_finding_reference_and_priorities_are_validated(validator):
    payload = assessment()
    payload["verdict"] = "request_changes"
    payload["obligation_assessments"][1].update(
        {
            "status": "partial",
            "validation_note": None,
            "gap": "Retry propagation is absent.",
        }
    )
    payload["repair_requests"] = [
        {
            "priority": 2,
            "finding_refs": ["O99"],
            "problem": "No propagation.",
            "requested_outcome": "Propagate it.",
            "suggested_validation": [],
        }
    ]
    with pytest.raises(ContractValidationError) as captured:
        validator.validate_assessment(
            payload,
            task_id="task-a",
            patch_sha256="a" * 64,
            ledger_sha256="b" * 64,
            ledger=ledger(),
        )
    assert "continuous from 1" in str(captured.value)
    assert "unknown finding O99" in str(captured.value)


def test_forbidden_evidence_markers_are_rejected(validator):
    payload = ledger()
    payload["obligations"][0]["origins"][0]["evidence"] = "tests/test.patch"
    with pytest.raises(ContractValidationError, match="forbidden evidence"):
        validator.validate_ledger(payload, task_id="task-a")


def test_schema_text_is_json_serializable_for_correction_prompt(validator):
    assert json.loads(json.dumps(validator.schema("assessment")))["$schema"].endswith(
        "2020-12/schema"
    )


@pytest.mark.parametrize(
    "replacement",
    (
        "Go template: {{.ITEM}}",
        "ANSI trace: {{0x01} {0x02 T 0x03}}",
        "Mixed {{if .Enabled}}template{{end}} output",
    ),
)
def test_prompt_renderer_does_not_rescan_inserted_double_braces(tmp_path, replacement):
    (tmp_path / "prompt.md").write_text("Issue:\n{{ISSUE}}\n")
    renderer = PromptRenderer(Path(tmp_path))

    rendered, _ = renderer.render("prompt.md", {"ISSUE": replacement})

    assert rendered == f"Issue:\n{replacement}\n"


def test_prompt_renderer_still_rejects_missing_original_template_replacement(tmp_path):
    (tmp_path / "prompt.md").write_text("{{KNOWN}} {{MISSING}}")
    renderer = PromptRenderer(Path(tmp_path))

    with pytest.raises(ValueError, match="MISSING"):
        renderer.render("prompt.md", {"KNOWN": "value with {{nested}} braces"})
