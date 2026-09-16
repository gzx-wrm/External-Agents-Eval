from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore
from external_agents_eval.workflows.solve_review_repair.models import (
    RoleName,
    RoleRunResult,
)
from external_agents_eval.workflows.solve_review_repair.v5_contracts import (
    NOTE_HEADINGS,
    V5ContractError,
    validate_output_schema,
    validate_review_batch_shape,
    validate_review_note,
    validate_spec_output,
    validate_summary_output,
)
from external_agents_eval.workflows.solve_review_repair.v5_controller import V5Controller
from external_agents_eval.workflows.solve_review_repair.v5_prompts import (
    REVIEW_BATCH_SCHEMA,
    SPEC_OUTPUT_SCHEMA,
    SUMMARY_OUTPUT_SCHEMA,
    correction_prompt,
    repair_prompt,
    reviewer_prompt,
    spec_prompt,
    summary_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = (
    ROOT
    / "external_agents_eval/workflows/solve_review_repair/skills/v5-review-note"
)
TASK_SOURCE = {
    "kind": "task",
    "field": "description",
    "quote": "Retry transient transport failures.",
}


def _spec_output() -> dict:
    return {
        "summary": "Add bounded retries.",
        "obligations": [
            {
                "statement": "Transient transport failures are retried.",
                "importance": "required",
                "sources": [TASK_SOURCE],
            }
        ],
        "preservation_obligations": [],
        "open_questions": [],
    }


def _review_batch() -> dict:
    return {
        "new_targets": [],
        "assessments": [
            {
                "target_id": "O001",
                "status": "violated",
                "expected": "Transient failures are retried.",
                "observed": "The first transient failure was returned.",
                "reason": "The observed public behavior contradicts the task.",
                "code_refs": [],
                "review_note_path": "notes/O001.md",
            }
        ],
        "revisit_target_ids": [],
        "done": True,
    }


def _summary_output() -> dict:
    finding_ref = "I:O001:v1"
    return {
        "summary": "Repair the confirmed retry violation.",
        "decisions": [
            {
                "finding_ref": finding_ref,
                "disposition": "repair",
                "reason": "The required behavior has a supported counterexample.",
                "supporting_refs": [],
                "sources": [TASK_SOURCE],
            }
        ],
        "repair_requests": [
            {
                "finding_refs": [finding_ref],
                "change": "Retry transient transport failures.",
                "validation_record_refs": [finding_ref],
                "preserve_refs": [],
            }
        ],
        "new_questions": [],
    }


def test_v5_output_schemas_are_valid_minimal_and_match_contract_validators():
    cases = (
        (SPEC_OUTPUT_SCHEMA, _spec_output()),
        (REVIEW_BATCH_SCHEMA, _review_batch()),
        (SUMMARY_OUTPUT_SCHEMA, _summary_output()),
    )
    for schema, payload in cases:
        Draft202012Validator.check_schema(schema)
        assert not list(Draft202012Validator(schema).iter_errors(payload))
        assert validate_output_schema(payload, schema) == payload

    assert validate_spec_output(_spec_output()) == _spec_output()
    assert validate_review_batch_shape(_review_batch()) == _review_batch()
    finding_ref = "I:O001:v1"
    assert validate_summary_output(
        _summary_output(),
        {finding_ref},
        decision_context={
            finding_ref: {
                "kind": "assessment",
                "status": "violated",
                "importance": "required",
            }
        },
        allowed_sources=[TASK_SOURCE],
    ) == _summary_output()

    for root_name, schema in (
        ("SpecOutput", SPEC_OUTPUT_SCHEMA),
        ("ReviewBatch", REVIEW_BATCH_SCHEMA),
        ("SummaryOutput", SUMMARY_OUTPUT_SCHEMA),
    ):
        assert schema["$ref"] == f"#/$defs/{root_name}"
        assert root_name in schema["$defs"]
    assert "SummaryOutput" not in SPEC_OUTPUT_SCHEMA["$defs"]
    assert "SpecOutput" not in REVIEW_BATCH_SCHEMA["$defs"]
    assert "ReviewBatch" not in SUMMARY_OUTPUT_SCHEMA["$defs"]


def test_v5_output_schema_rejects_missing_wrong_type_and_extra_fields():
    missing = _review_batch()
    del missing["done"]
    with pytest.raises(V5ContractError, match=r"\$.*done.*required"):
        validate_output_schema(missing, REVIEW_BATCH_SCHEMA)

    wrong_type = _review_batch()
    wrong_type["done"] = "yes"
    with pytest.raises(V5ContractError, match=r"\$\.done"):
        validate_output_schema(wrong_type, REVIEW_BATCH_SCHEMA)

    extra = _spec_output()
    extra["verdict"] = "pass"
    with pytest.raises(V5ContractError, match="Additional properties"):
        validate_output_schema(extra, SPEC_OUTPUT_SCHEMA)


def test_v5_prompts_render_role_boundaries_and_current_contracts():
    task = {
        "title": "Retry requests",
        "description": "Retry transient transport failures.",
    }
    repository = {
        "base_root": "/work/base",
        "candidate_root": "/work/candidate",
        "diff_path": "/work/input/p0.patch",
        "scratch_root": "/review-scratch",
    }
    ledger = {
        "summary": "Add retries.",
        "obligations": [{"id": "O001", "content": _spec_output()["obligations"][0]}],
        "preservation_obligations": [],
        "open_questions": [],
    }
    state = {
        "mode": "continue",
        "pending": ["O001"],
        "review_notes": {
            "skill_path": "/skills/v5-review-note/SKILL.md",
            "output_root": "/review-scratch",
            "target_paths": {"O001": "notes/O001.md"},
        },
    }

    specification = spec_prompt(task, repository)
    implementation = reviewer_prompt(
        task=task,
        repository=repository,
        ledger=ledger,
        state=state,
        role="I",
        skill_path="/skills/v5-review-note/SKILL.md",
        output_root="/review-scratch",
    )
    regression = reviewer_prompt(
        task=task,
        repository=repository,
        ledger=ledger,
        state=state,
        role="R",
        skill_path="/skills/v5-review-note/SKILL.md",
        output_root="/review-scratch",
    )
    summary = summary_prompt(
        {
            "task": task,
            "ledger": ledger,
            "reviews": {"I": {}, "R": {}},
            "required_decision_refs": [],
            "allowed_supporting_refs": [],
            "allowed_validation_record_refs": [],
            "allowed_preserve_refs": [],
            "allowed_sources": [],
            "review_notes": [],
            "artifact_root": "/artifacts",
        },
        freeze_id="freeze-1",
        source_bindings=[TASK_SOURCE],
    )
    repair = repair_prompt(task, "a" * 64, _summary_output())
    corrector = correction_prompt(
        original_output="{}",
        output_schema=REVIEW_BATCH_SCHEMA,
        errors=[{"path": "$.done", "message": "required"}],
    )

    assert '"diff_path"' not in specification
    assert '"candidate_root"' not in specification
    assert '"$ref": "#/$defs/SpecOutput"' in specification
    assert "candidate diff path is deliberately absent" in specification

    for prompt in (implementation, regression):
        assert "synthetic examples only as writing guidance" in prompt
        assert "write and read back one Markdown note" in prompt
        assert "resulting pending set is empty" in prompt
        assert "len(new_targets) + len(assessments)" in prompt
        assert "limit counts all three arrays" in prompt
        assert '"$ref": "#/$defs/ReviewBatch"' in prompt
        assert "Do not use the network" in prompt
        assert "representative success case" in prompt
        assert "boundary case" in prompt
        assert "invalid or negative case" in prompt
        assert "normal public entry point" in prompt
        assert "raw observable result" in prompt
        assert "silently changing the call or adapting the result" in prompt
        assert "Any contrary or" in prompt
    assert "I must cover every O target" in implementation
    assert "ledger as a fallible checklist" in implementation
    assert "independently of candidate tests" in implementation
    assert "R must cover\nevery P/R target" in regression
    assert "Work independently of I" in regression
    assert "failure also present at base is not by\nitself a regression" in regression

    neutral_lines = (
        "Adjudicate I/R disagreements independently from the task, applicable contracts,",
        "and supplied evidence, without favoring either reviewer.",
    )
    assert all(line in summary for line in neutral_lines)
    assert "Read\nevery non-satisfied assessment's note" in summary
    assert "trace/material manifests" in summary
    assert "allowed_* catalogs" in summary
    assert "never bare Q001/O001/P001" in summary
    assert "never belong in supporting_refs" in summary
    assert "Incomplete reports cannot support acceptance" in summary
    assert "Never access hidden tests" in summary
    assert '"$ref": "#/$defs/SummaryOutput"' in summary

    assert "no additional structured output contract" in repair
    assert "OUTPUT JSON SCHEMA" not in repair
    assert "syntax or shape defects only" in corrector
    assert "Do not invent missing evidence" in corrector


def test_v5_review_note_skill_links_complete_synthetic_examples_without_leakage(
    tmp_path,
):
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", skill)
    assert links == [
        "assets/review-note-template.md",
        "references/static-satisfied.md",
        "references/dynamic-violated.md",
        "references/regression-violated.md",
        "references/blocked-unresolved.md",
        "references/source-disputed.md",
    ]
    for relative in links:
        path = (SKILL_ROOT / relative).resolve()
        path.relative_to(SKILL_ROOT.resolve())
        assert path.is_file()
    assert "Examples are synthetic writing examples" in skill
    assert "All four statuses require notes" in skill
    assert "Use `disputed`" in skill and "Use `unresolved`" in skill

    examples = {
        "P001": "references/static-satisfied.md",
        "O001": "references/dynamic-violated.md",
        "P002": "references/regression-violated.md",
        "O002": "references/blocked-unresolved.md",
        "O003": "references/source-disputed.md",
    }
    combined = skill
    for target, relative in examples.items():
        content = (SKILL_ROOT / relative).read_text(encoding="utf-8")
        combined += content
        destination = tmp_path / "notes" / f"{target}.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        assert validate_review_note(tmp_path, f"notes/{target}.md", target)

    template = (SKILL_ROOT / "assets/review-note-template.md").read_text(
        encoding="utf-8"
    )
    assert "{{TARGET_ID}}" in template
    assert all(template.count(heading) == 1 for heading in NOTE_HEADINGS)

    manifest = json.loads(
        (ROOT / "manifests/deepswe_v1.1_pi_random60_p0_20260901.json").read_text(
            encoding="utf-8"
        )
    )
    assert not [task_id for task_id in manifest["task_ids"] if task_id in combined]


class _ShapeCorrectionRunner:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.calls: list[dict] = []

    async def run_role(self, **kwargs):
        self.calls.append(kwargs)
        trajectory = self.tmp_path / "corrector.json"
        trajectory.write_text("{}", encoding="utf-8")
        return RoleRunResult(
            exit_code=0,
            final_assistant_text=json.dumps(_review_batch()),
            transcript_path=trajectory,
            trajectory_path=trajectory,
            started_at="now",
            finished_at="now",
        )


def test_v5_schema_shape_failure_uses_tool_free_format_corrector(tmp_path):
    runner = _ShapeCorrectionRunner(tmp_path)
    controller = V5Controller(
        role_runner=runner,
        workspace=object(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    try:
        result = asyncio.run(
            controller._parse_with_format_corrections(
                raw='{"new_targets": [], "assessments": [], "revisit_target_ids": []}',
                output_schema=REVIEW_BATCH_SCHEMA,
                worktree=str(tmp_path),
                prefix="readiness",
            )
        )
    finally:
        controller.close()

    assert result == _review_batch()
    assert len(runner.calls) == 1
    assert runner.calls[0]["role"] == RoleName.JSON_CORRECTOR
    assert runner.calls[0]["tools"] == ()
    assert "JSON Schema validation failed" in runner.calls[0]["prompt"]
