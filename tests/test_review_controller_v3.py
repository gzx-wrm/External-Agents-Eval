import asyncio
import copy
import json
from pathlib import Path

import pytest

from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore, sha256_bytes
from external_agents_eval.workflows.solve_review_repair.config import WorkflowConfig
from external_agents_eval.workflows.solve_review_repair.controller import (
    SolveReviewRepairController,
)
from external_agents_eval.workflows.solve_review_repair.models import (
    ExistingPatch,
    RoleName,
    RoleRunResult,
    TaskInput,
    WorkflowState,
    WorkflowWorktrees,
)
from external_agents_eval.workflows.solve_review_repair.validators import (
    ContractValidationError,
)

P0 = b"""diff --git a/src/client.py b/src/client.py
--- a/src/client.py
+++ b/src/client.py
@@ -1 +1 @@
-timeout = None
+timeout = 10
"""
P1 = b"""diff --git a/src/client.py b/src/client.py
--- a/src/client.py
+++ b/src/client.py
@@ -1 +1 @@
-timeout = None
+timeout = 20
"""


class V3FakeWorkspace:
    def __init__(self):
        self.contents = {}
        self.untracked = {}
        self.materialized = {}
        self.published = None
        self.cleaned = False

    async def prepare(self, *, base_commit):
        self.base_commit = base_commit
        paths = WorkflowWorktrees("solver", "spec", "patch", "repair", "correction", "final")
        self.contents = {path: b"" for path in paths.__dict__.values()}
        self.untracked = {path: b"" for path in paths.__dict__.values()}
        return paths

    async def apply_patch(self, *, worktree, patch):
        self.contents[worktree] = patch

    async def discard_worktree(self, *, worktree):
        self.contents[worktree] = b""

    async def materialize_file(self, *, name, payload, readonly=False):
        path = f"/contracts/{name}"
        self.materialized[path] = payload
        return path

    async def materialized_file_digest(self, *, path):
        return sha256_bytes(self.materialized[path])

    async def read_materialized_file(self, *, path):
        return self.materialized[path]

    async def capture_patch(self, *, worktree, base_commit):
        return self.contents[worktree]

    async def tracked_diff_digest(self, *, worktree, base_commit):
        return sha256_bytes(self.contents[worktree] + self.untracked[worktree])

    async def tracked_files_diff_digest(self, *, worktree, base_commit):
        return sha256_bytes(self.contents[worktree])

    async def restore_reviewer_worktree(self, *, worktree, base_commit):
        discarded = self.untracked[worktree]
        self.untracked[worktree] = b""
        if worktree == "spec":
            self.contents[worktree] = b""
        return discarded

    async def validate_evidence(self, *, worktree, assessment):
        return []

    async def validate_repository_sources(self, *, worktree, base_commit, sources):
        raise AssertionError("V3 must use deterministic source binding")

    async def bind_repository_sources(self, *, worktree, base_commit, sources):
        return [dict(source) for source in sources], []

    async def publish_final_patch(self, *, base_commit, patch):
        self.published = patch

    async def restore_final_worktree(self, *, base_commit):
        discarded = self.contents["final"]
        self.contents["final"] = b""
        return discarded

    async def cleanup(self):
        self.cleaned = True


class V3FakeRoleRunner:
    def __init__(self, workspace):
        self.workspace = workspace
        self.calls = []
        self.prompts = {}

    async def run_role(
        self,
        *,
        role,
        prompt,
        worktree,
        tools,
        timeout_sec,
        artifact_prefix,
    ):
        self.calls.append(role)
        self.prompts[role] = prompt
        if role == RoleName.SPECIFICATION_REVIEWER:
            build_input = json.loads(
                prompt.split("TASK INPUT JSON\n", 1)[1].split("\n\nOUTPUT CONTRACT", 1)[0]
            )
            payload = {
                "schema_version": "3.0",
                "task_id": build_input["task_id"],
                "summary": "Accept and apply an explicit timeout.",
                "sources": [
                    {
                        "id": "S01",
                        "kind": "task",
                        "field": "description",
                        "quote": "Accept a timeout",
                    }
                ],
                "obligations": [
                    {
                        "id": "O01",
                        "statement": "The public client applies the explicit timeout.",
                        "criticality": "must",
                        "basis": "explicit_task",
                        "source_refs": ["S01"],
                        "rationale": "S01 directly requires the timeout behavior.",
                    }
                ],
                "open_questions": [],
            }
            text = "```json\n" + json.dumps(payload) + "\n```"
        elif role == RoleName.PATCH_REVIEWER:
            assess_input = json.loads(
                prompt.split("TASK INPUT JSON\n", 1)[1].split("\n\nOUTPUT CONTRACT", 1)[0]
            )
            payload = {
                "schema_version": "3.0",
                "task_id": assess_input["task_id"],
                "patch_sha256": assess_input["patch"]["sha256"],
                "ledger_sha256": assess_input["obligation_ledger"]["sha256"],
                "additional_sources": [],
                "late_obligations": [],
                "obligation_assessments": [
                    {
                        "obligation_ref": "O01",
                        "source_status": "supported",
                        "source_note": None,
                        "implementation_status": "missing",
                        "implementation_evidence": [],
                        "validation_evidence": [],
                        "gap": "The candidate keeps the previous timeout value.",
                    }
                ],
                "unexpected_regressions": [],
                "repair_requests": [
                    {
                        "id": "Q01",
                        "finding_refs": ["O01"],
                        "action": "fix",
                        "problem": "The explicit timeout is not applied.",
                        "required_outcome": "Apply the explicit timeout.",
                        "preserve": ["The module remains importable."],
                        "validation": ["Read the public timeout value."],
                    }
                ],
                "verdict": "request_changes",
                "verdict_rationale": "One localized must gap remains.",
            }
            text = "```json\n" + json.dumps(payload) + "\n```"
        elif role == RoleName.REPAIR:
            repair_input = json.loads(prompt.split("REPAIR INPUT JSON\n", 1)[1])
            self.workspace.contents[worktree] = P1
            audit = repair_input["audit"]
            self.workspace.materialized[audit["path"]] = json.dumps(
                {
                    "schema_version": "3.0",
                    "task_id": repair_input["task_id"],
                    "candidate_patch_sha256": repair_input["candidate_patch"]["sha256"],
                    "assessment_sha256": audit["assessment_sha256"],
                    "effective_review_set_sha256": audit[
                        "effective_review_set_sha256"
                    ],
                    "changed_paths": ["src/client.py"],
                    "changes": [
                        {
                            "change_id": "C01",
                            "request_refs": ["Q01"],
                            "finding_refs": ["O01"],
                            "decision": "implement",
                            "before_behavior": "The explicit timeout was not applied.",
                            "after_behavior": "The explicit timeout is applied.",
                            "preserve": ["The module remains importable."],
                            "out_of_request_scope": False,
                            "scope_rationale": None,
                            "changed_paths": ["src/client.py"],
                            "validation": [
                                {
                                    "command": "python -c 'import client'",
                                    "result": "passed",
                                    "expected": "The module imports with the repaired value.",
                                    "actual": "The module imported with the repaired value.",
                                    "log_path": None,
                                }
                            ],
                        }
                    ],
                    "unaddressed_requests": [],
                    "validation_summary": "The focused public import passed.",
                }
            ).encode()
            text = "repair complete"
        else:
            raise AssertionError(f"Unexpected role: {role}")
        return RoleRunResult(
            exit_code=0,
            final_assistant_text=text,
            transcript_path=Path("transcript"),
            trajectory_path=Path("trajectory"),
            started_at="start",
            finished_at="finish",
        )


class V3CorrectionRunner:
    def __init__(self, corrected):
        self.corrected = corrected
        self.calls = []

    async def run_role(
        self,
        *,
        role,
        prompt,
        worktree,
        tools,
        timeout_sec,
        artifact_prefix,
    ):
        assert role == RoleName.JSON_CORRECTOR
        self.calls.append(role)
        return RoleRunResult(
            exit_code=0,
            final_assistant_text="```json\n" + json.dumps(self.corrected) + "\n```",
            transcript_path=Path("transcript"),
            trajectory_path=Path("trajectory"),
            started_at="start",
            finished_at="finish",
        )


def correction_assessment_v3():
    return {
        "schema_version": "3.0",
        "task_id": "task-a",
        "patch_sha256": "1" * 64,
        "ledger_sha256": "2" * 64,
        "additional_sources": [
            {
                "id": "A01",
                "kind": "repository",
                "path": "src/client.py",
                "line_start": 1,
                "line_end": 1,
            }
        ],
        "late_obligations": [],
        "obligation_assessments": [
            {
                "obligation_ref": "O01",
                "source_status": "supported",
                "source_note": None,
                "implementation_status": "covered",
                "implementation_evidence": [
                    {
                        "path": "src/client.py",
                        "line_start": 1,
                        "line_end": 1,
                        "explanation": "The public client applies the timeout.",
                    }
                ],
                "validation_evidence": [],
                "gap": None,
            }
        ],
        "unexpected_regressions": [],
        "repair_requests": [],
        "verdict": "approve",
        "verdict_rationale": "The supported must obligation is covered.",
    }


def correction_controller_v3(tmp_path, role_runner=None):
    workspace = V3FakeWorkspace()
    roles = role_runner or V3FakeRoleRunner(workspace)
    return SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        config=WorkflowConfig(
            contract_version="3.0",
            experiment_arm="E2",
            max_obligations=64,
            prompt_version="4.0",
            controller_version="3.0.1",
        ),
    )


def test_v3_fake_full_chain_binds_sources_expands_requests_and_validates_audit(tmp_path):
    p0_path = tmp_path / "p0.patch"
    p0_path.write_bytes(P0)
    existing = ExistingPatch(
        task_id="task-a",
        base_commit="a" * 40,
        path=p0_path,
        sha256=sha256_bytes(P0),
        tracked_diff_sha256=sha256_bytes(P0),
        source_job_id="job-a",
        source_trial_id="task-a__p0",
    )
    task = TaskInput(
        task_id="task-a",
        issue_title="Timeout",
        issue_body="Accept a timeout and apply it to the public client.",
        repository_name="fixture",
        base_commit="a" * 40,
    )
    workspace = V3FakeWorkspace()
    roles = V3FakeRoleRunner(workspace)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
        config=WorkflowConfig(
            contract_version="3.0",
            experiment_arm="E2",
            max_obligations=64,
            prompt_version="3.0",
            controller_version="3.0.1",
        ),
    )

    result = asyncio.run(controller.run(task, existing_p0=existing))

    assert result.state == WorkflowState.COMPLETED
    assert result.p1_patch == P1
    assert workspace.published == P1
    assert workspace.cleaned is True
    assert roles.calls == [
        RoleName.SPECIFICATION_REVIEWER,
        RoleName.PATCH_REVIEWER,
        RoleName.REPAIR,
    ]
    source_bindings = json.loads(
        (artifacts.root / "reviewer/source_bindings.json").read_text()
    )
    assert source_bindings["sources"] == [
        {
            "field": "description",
            "id": "S01",
            "kind": "task",
            "quote": "Accept a timeout",
        }
    ]
    additional_bindings = json.loads(
        (artifacts.root / "reviewer/additional_source_bindings.json").read_text()
    )
    assert additional_bindings["sources"] == []
    repair_input = json.loads((artifacts.root / "inputs/repair_input.json").read_text())
    assert repair_input["effective_review_set"]["effective_obligations"] == ["O01"]
    assert repair_input["request_contexts"][0]["order"] == 1
    assert repair_input["request_contexts"][0]["source_ids"] == ["S01"]
    assert repair_input["request_contexts"][0]["obligations"][0]["id"] == "O01"
    assert repair_input["source_bindings"]["sources"] == source_bindings["sources"]
    assert result.repair_audit["changes"][0]["finding_refs"] == ["O01"]
    assert '"title": "ReviewObligationsV3"' in roles.prompts[
        RoleName.SPECIFICATION_REVIEWER
    ]
    assert '"title": "PatchAssessmentV3"' in roles.prompts[RoleName.PATCH_REVIEWER]
    assert "criterion_ref" not in roles.prompts[RoleName.REPAIR]
    manifest = json.loads((artifacts.root / "manifest.json").read_text())
    assert manifest["contract_version"] == "3.0"
    assert manifest["source_bindings_sha256"]
    assert manifest["schema_summary"]["contract_version"] == "3.0"
    assert manifest["repair_audit_status"] == "validated"


@pytest.mark.parametrize(
    "malformation",
    ("repository_quote", "missing_neutral_gap", "unknown_fields"),
)
def test_v3_correction_semantics_normalizes_schema_only_changes(
    tmp_path, malformation
):
    controller = correction_controller_v3(tmp_path)
    corrected = correction_assessment_v3()
    malformed = copy.deepcopy(corrected)
    if malformation == "repository_quote":
        malformed["additional_sources"][0]["quote"] = "timeout = 10"
    elif malformation == "missing_neutral_gap":
        del malformed["obligation_assessments"][0]["gap"]
    else:
        malformed["no_extra_field"] = None
        malformed["obligation_assessments"][0]["implementation_evidence"][0][
            "line_end2"
        ] = 0

    controller.validator.validate_schema("assessment", corrected)
    with pytest.raises(ContractValidationError):
        controller.validator.validate_schema("assessment", malformed)
    assert controller._correction_semantics_sha256(
        "assessment", malformed
    ) == controller._correction_semantics_sha256("assessment", corrected)


@pytest.mark.parametrize(
    "mutation",
    ("verdict", "status", "evidence_line", "evidence_explanation", "gap_text"),
)
def test_v3_correction_semantics_still_protects_reviewer_decisions(tmp_path, mutation):
    controller = correction_controller_v3(tmp_path)
    original = correction_assessment_v3()
    changed = copy.deepcopy(original)
    assessment = changed["obligation_assessments"][0]
    if mutation == "verdict":
        changed["verdict"] = "request_changes"
    elif mutation == "status":
        assessment["implementation_status"] = "partial"
    elif mutation == "evidence_line":
        assessment["implementation_evidence"][0]["line_end"] = 2
    elif mutation == "evidence_explanation":
        assessment["implementation_evidence"][0]["explanation"] = "Changed evidence."
    else:
        assessment["gap"] = "A newly asserted semantic gap."

    assert controller._correction_semantics_sha256(
        "assessment", original
    ) != controller._correction_semantics_sha256("assessment", changed)


def test_v3_schema_only_correction_resumes_validation(tmp_path):
    corrected = correction_assessment_v3()
    malformed = copy.deepcopy(corrected)
    malformed["additional_sources"][0]["quote"] = "timeout = 10"
    del malformed["obligation_assessments"][0]["gap"]
    malformed["no_extra_field"] = None
    malformed["obligation_assessments"][0]["implementation_evidence"][0][
        "line_end2"
    ] = 0
    roles = V3CorrectionRunner(corrected)
    controller = correction_controller_v3(tmp_path, roles)

    payload = asyncio.run(
        controller._validate_with_corrections(
            raw="```json\n" + json.dumps(malformed) + "\n```",
            schema_name="assessment",
            semantic_validate=lambda item: controller.validator.validate_schema(
                "assessment", item
            ),
            worktree="correction",
            prefix="reviewer/stage_b_correction",
        )
    )

    assert payload == corrected
    assert roles.calls == [RoleName.JSON_CORRECTOR]
    validation = json.loads(
        (controller.artifacts.root / "reviewer/stage_b_validation.json").read_text()
    )
    assert validation["valid"] is True
    assert validation["correction_used"] is True
    assert validation["correction_count"] == 1
