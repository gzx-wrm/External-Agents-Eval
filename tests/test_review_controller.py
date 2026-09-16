import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from external_agents_eval.workflows.solve_review_repair.artifacts import (
    ArtifactStore,
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.config import WorkflowConfig
from external_agents_eval.workflows.solve_review_repair.controller import (
    SolveReviewRepairController,
    git_commits_equivalent,
)
from external_agents_eval.workflows.solve_review_repair.models import (
    ExistingPatch,
    RoleName,
    RoleRunResult,
    TaskInput,
    WorkflowState,
    WorkflowWorktrees,
)


def valid_ledger():
    return {
        "schema_version": "1.0",
        "task_id": "task-a",
        "feature_summary": "Expose a timeout.",
        "scope": {"in_scope": ["client"], "out_of_scope": [], "assumptions": []},
        "obligations": [
            {
                "id": "O01",
                "statement": "The client accepts a timeout.",
                "origins": [
                    {
                        "type": "explicit_issue",
                        "evidence": "accept timeout",
                        "rationale": "Explicit requirement.",
                    }
                ],
                "category": "public_api",
                "criticality": "must",
                "acceptance_criteria": ["A timeout is accepted."],
                "affected_surfaces": ["Client"],
                "dependencies": [],
                "confidence": "high",
            }
        ],
    }


def valid_assessment(assess_input, verdict="approve"):
    status = "covered" if verdict == "approve" else "partial"
    payload = {
        "schema_version": "1.0",
        "task_id": "task-a",
        "patch_sha256": assess_input["patch"]["sha256"],
        "ledger_sha256": assess_input["obligation_ledger"]["sha256"],
        "obligation_assessments": [
            {
                "obligation_id": "O01",
                "status": status,
                "implementation_evidence": [
                    {
                        "path": "src/client.py",
                        "symbol": "Client",
                        "line_start": 1,
                        "line_end": 1,
                        "explanation": "The implementation is here.",
                    }
                ],
                "test_evidence": [],
                "validation_note": "Inspected directly." if status == "covered" else None,
                "gap": None if status == "covered" else "Retry propagation is absent.",
                "confidence": "high",
            }
        ],
        "unexpected_regressions": [],
        "late_obligation_candidates": [],
        "verdict": verdict,
        "verdict_rationale": "Covered." if verdict == "approve" else "A local gap remains.",
        "repair_requests": [],
    }
    if verdict != "approve":
        payload["repair_requests"] = [
            {
                "priority": 1,
                "finding_refs": ["O01"],
                "problem": "Retry propagation is absent.",
                "requested_outcome": "Apply timeout to retries.",
                "suggested_validation": ["Exercise a retry."],
            }
        ]
    return payload


def valid_ledger_v2(source_policy_sha):
    return {
        "schema_version": "2.0",
        "task_id": "task-a",
        "source_policy_sha256": source_policy_sha,
        "feature_summary": "Expose a timeout.",
        "scope": {"in_scope": ["client"], "out_of_scope": [], "assumptions": []},
        "sources": [
            {
                "id": "S01",
                "kind": "issue",
                "locator": {"field": "body", "quote": "Accept a timeout"},
            }
        ],
        "obligations": [
            {
                "id": "O01",
                "statement": "The client accepts a timeout.",
                "category": "public_api",
                "acceptance_criteria": [
                    {
                        "id": "O01.C01",
                        "statement": "Construction accepts a timeout.",
                        "criticality": "must",
                        "basis_kind": "explicit_issue",
                        "source_refs": ["S01"],
                        "derivation": "The issue explicitly requests the input.",
                        "open_questions": [],
                        "confidence": "high",
                    }
                ],
                "affected_surfaces": ["Client"],
                "dependencies": [],
            }
        ],
    }


def valid_assessment_v2(assess_input, verdict="approve"):
    status = {
        "approve": "covered",
        "abstain": "uncertain",
        "request_changes": "missing",
        "replan": "missing",
    }[verdict]
    payload = {
        "schema_version": "2.0",
        "task_id": "task-a",
        "patch_sha256": assess_input["patch"]["sha256"],
        "ledger_sha256": assess_input["obligation_ledger"]["sha256"],
        "additional_sources": [],
        "criterion_assessments": [
            {
                "criterion_ref": "O01.C01",
                "source_check": {
                    "status": "supported",
                    "source_refs": ["S01"],
                    "rationale": "The issue establishes this behavior.",
                    "compliant_but_rejected_example": None,
                    "satisfies_but_violates_example": None,
                },
                "implementation_status": status,
                "implementation_evidence": [
                    {
                        "path": "src/client.py",
                        "symbol": "Client",
                        "line_start": 1,
                        "line_end": 1,
                        "explanation": "The constructor is implemented here.",
                    }
                ],
                "test_evidence": [],
                "validation_note": "Inspected directly." if status == "covered" else None,
                "gap": None if status == "covered" else "The behavior is not established.",
                "confidence": "high",
            }
        ],
        "ledger_challenges": [],
        "unexpected_regressions": [],
        "late_obligation_candidates": [],
        "verdict": verdict,
        "verdict_rationale": "The deterministic v2 route was selected.",
        "abstain_reason": (
            "The required runtime is unavailable." if verdict == "abstain" else None
        ),
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
                "action_kind": "fix",
                "problem": "The timeout is not accepted.",
                "requested_outcome": "Accept the timeout.",
                "preserve_behaviors": ["Calls without a timeout remain valid."],
                "suggested_validation": ["Exercise both constructor forms."],
            }
        ]
    return payload


def artifact_json_sha(payload):
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    return sha256_bytes(rendered.encode())


class FakeWorkspace:
    def __init__(
        self,
        p0=b"P0\n",
        p1=b"P1\n",
        evidence_errors=None,
        initial_final=b"",
    ):
        self.p0 = p0
        self.p1 = p1
        self.contents = {}
        self.published = None
        self.cleaned = False
        self.materialized = {}
        self.evidence_errors = list(evidence_errors or [])
        self.initial_final = initial_final
        self.restored = []
        self.untracked = {}

    async def prepare(self, *, base_commit):
        self.base_commit = base_commit
        paths = WorkflowWorktrees("solver", "spec", "patch", "repair", "correction", "final")
        self.contents = {path: b"" for path in paths.__dict__.values()}
        self.untracked = {path: b"" for path in paths.__dict__.values()}
        self.contents[paths.final] = self.initial_final
        return paths

    async def apply_patch(self, *, worktree, patch):
        self.contents[worktree] = patch

    async def discard_worktree(self, *, worktree):
        self.contents[worktree] = b""

    async def materialize_file(self, *, name, payload, readonly=False):
        self.materialized[f"/contracts/{name}"] = payload
        return f"/contracts/{name}"

    async def materialized_file_digest(self, *, path):
        return sha256_bytes(self.materialized[path])

    async def read_materialized_file(self, *, path):
        return self.materialized[path]

    async def capture_patch(self, *, worktree, base_commit):
        if worktree == "solver":
            return self.p0
        return self.contents[worktree]

    async def tracked_diff_digest(self, *, worktree, base_commit):
        return sha256_bytes(self.contents[worktree] + self.untracked[worktree])

    async def tracked_files_diff_digest(self, *, worktree, base_commit):
        return sha256_bytes(self.contents[worktree])

    async def restore_reviewer_worktree(self, *, worktree, base_commit):
        discarded = self.contents[worktree] + self.untracked[worktree]
        self.contents[worktree] = b""
        self.untracked[worktree] = b""
        return discarded

    async def validate_evidence(self, *, worktree, assessment):
        return self.evidence_errors.pop(0) if self.evidence_errors else []

    async def validate_repository_sources(self, *, worktree, base_commit, sources):
        return []

    async def publish_final_patch(self, *, base_commit, patch):
        self.published = patch

    async def restore_final_worktree(self, *, base_commit):
        discarded = self.contents["final"]
        self.contents["final"] = b""
        self.restored.append((base_commit, discarded))
        return discarded

    async def cleanup(self):
        self.cleaned = True


class FakeRoleRunner:
    def __init__(self, workspace, *, verdict="approve"):
        self.workspace = workspace
        self.verdict = verdict
        self.calls = []
        self.overrides = {}
        self.corrections = []
        self.mutate_role = None
        self.fence_roles = set()
        self.timeouts = []
        self.write_repair_audit = True

    async def run_role(self, *, role, prompt, worktree, tools, timeout_sec, artifact_prefix):
        self.calls.append(role)
        self.timeouts.append((role, timeout_sec))
        if role == self.mutate_role:
            self.workspace.contents[worktree] += b"mutation\n"
        if role == RoleName.SPECIFICATION_REVIEWER:
            assert self.workspace.contents["solver"] == b""
            assert self.workspace.contents["patch"] == b""
            assert self.workspace.contents["repair"] == b""
            assert self.workspace.contents["final"] == b""
            if "schema version 2.0" in prompt:
                build_input_text = prompt.split("TASK INPUT JSON\n", 1)[1]
                build_input = json.loads(
                    build_input_text.split("\n\nFINAL RESPONSE REMINDER", 1)[0]
                )
                text = json.dumps(valid_ledger_v2(build_input["source_policy"]["sha256"]))
            else:
                text = json.dumps(valid_ledger())
        elif role == RoleName.PATCH_REVIEWER:
            assert self.workspace.contents["patch"] == self.workspace.p0
            assert self.workspace.contents["repair"] == b""
            assert self.workspace.contents["final"] == b""
            assess_input_text = prompt.split("TASK INPUT JSON\n", 1)[1]
            assess_input = json.loads(assess_input_text.split("\n\nFINAL RESPONSE REMINDER", 1)[0])
            if assess_input["schema_version"] == "2.0":
                text = json.dumps(valid_assessment_v2(assess_input, self.verdict))
            else:
                text = json.dumps(valid_assessment(assess_input, self.verdict))
        elif role == RoleName.JSON_CORRECTOR:
            text = self.corrections.pop(0)
        elif role == RoleName.REPAIR:
            self.workspace.contents[worktree] = self.workspace.p1
            if '"schema_version": "2.0"' in prompt:
                repair_input = json.loads(prompt.split("REPAIR INPUT JSON\n", 1)[1])
                audit = repair_input["audit"]
                if audit["required"] and self.write_repair_audit:
                    request_ids = [
                        request["id"]
                        for request in repair_input["patch_assessment"]["repair_requests"]
                    ]
                    payload = {
                        "schema_version": "2.0",
                        "task_id": "task-a",
                        "p0_sha256": repair_input["patch"]["sha256"],
                        "assessment_sha256": audit["assessment_sha256"],
                        "effective_review_set_sha256": audit["effective_review_set_sha256"],
                        "additional_sources": [],
                        "changed_paths": [],
                        "changes": [
                            {
                                "change_id": "C01",
                                "request_refs": request_ids,
                                "criterion_refs": ["O01.C01"],
                                "risk_tags": ["public_api"],
                                "before_behavior": "P0 does not accept the timeout.",
                                "after_behavior": "The proposed semantic change was rejected.",
                                "source_refs": ["S01"],
                                "source_check": "contradicted",
                                "counterevidence": ["The fake audit exercises rejection."],
                                "preserve_behaviors": [],
                                "decision": "reject",
                                "changed_paths": [],
                                "validation": [],
                            }
                        ],
                        "unaddressed_requests": [],
                        "validation_summary": "The fake workflow recorded every request.",
                    }
                    self.workspace.materialized[audit["path"]] = json.dumps(payload).encode()
            text = "done"
        else:
            text = "solved"
        if role in self.fence_roles:
            text = f"Here is the requested JSON.\n```json\n{text}\n```"
        result = RoleRunResult(
            exit_code=0,
            final_assistant_text=text,
            transcript_path=Path("transcript"),
            trajectory_path=Path("trajectory"),
            started_at="start",
            finished_at="finish",
        )
        override = self.overrides.get(role)
        return replace(result, **override) if override else result


def task():
    return TaskInput(
        task_id="task-a",
        issue_title="Timeout",
        issue_body='Accept a timeout containing JSON-sensitive "quotes".',
        repository_name="repo",
        base_commit="a" * 40,
    )


def existing_p0(tmp_path, patch=b"P0\n"):
    path = tmp_path / "p0.patch"
    path.write_bytes(patch)
    digest = sha256_bytes(patch)
    return ExistingPatch(
        task_id="task-a",
        base_commit="a" * 40,
        path=path,
        sha256=digest,
        tracked_diff_sha256=digest,
        source_job_id="job-1",
        source_trial_id="task-a__trial",
    )


def run_controller(tmp_path, *, verdict="approve", configure=None, config=None, workspace=None):
    workspace = workspace or FakeWorkspace()
    roles = FakeRoleRunner(workspace, verdict=verdict)
    if configure:
        configure(roles)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
        config=config,
    )
    result = asyncio.run(controller.run(task(), existing_p0=existing_p0(tmp_path)))
    return result, roles, workspace, artifacts


def test_approve_skips_repair_and_preserves_p0(tmp_path):
    result, roles, workspace, artifacts = run_controller(tmp_path)
    assert result.state == WorkflowState.COMPLETED
    assert result.repair_executed is False
    assert RoleName.REPAIR not in roles.calls
    assert workspace.published == b"P0\n"
    assert artifacts.digest_file("outputs/p1.patch") == result.p1_sha256
    usage = json.loads((artifacts.root / "outputs/resource_usage.json").read_text())
    assert [item["role"] for item in usage["roles"]] == [
        RoleName.SPECIFICATION_REVIEWER.value,
        RoleName.PATCH_REVIEWER.value,
    ]
    assert workspace.cleaned is True


def test_request_changes_and_replan_each_repair_exactly_once(tmp_path):
    for index, verdict in enumerate(("request_changes", "replan")):
        result, roles, workspace, _ = run_controller(tmp_path / str(index), verdict=verdict)
        assert result.state == WorkflowState.COMPLETED
        assert result.repair_executed is True
        assert roles.calls.count(RoleName.REPAIR) == 1
        assert workspace.published == b"P1\n"


def test_repair_timeout_preserves_usable_partial_diff_and_stays_single_pass(tmp_path):
    def configure(roles):
        roles.overrides[RoleName.REPAIR] = {
            "exit_code": 124,
            "timed_out": True,
            "exception": "timeout",
        }

    result, roles, workspace, _ = run_controller(
        tmp_path,
        verdict="request_changes",
        configure=configure,
    )
    assert result.state == WorkflowState.REPAIR_FAILED
    assert roles.calls.count(RoleName.REPAIR) == 1
    assert workspace.published == b"P1\n"
    assert result.error == "timeout"


def test_solver_timeout_publishes_partial_patch_for_verification(tmp_path):
    workspace = FakeWorkspace(p0=b"partial solver patch\n")
    roles = FakeRoleRunner(workspace)
    roles.overrides[RoleName.SOLVER] = {
        "exit_code": 124,
        "timed_out": True,
        "exception": "timeout",
    }
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
    )

    result = asyncio.run(controller.run(task()))

    assert result.state == WorkflowState.SOLVER_FAILED
    assert result.p1_patch == b"partial solver patch\n"
    assert workspace.published == b"partial solver patch\n"
    assert roles.calls == [RoleName.SOLVER]
    assert roles.timeouts == [(RoleName.SOLVER, 10_800)]
    assert (artifacts.root / "manifest.json").is_file()
    assert artifacts.digest_file("outputs/p1.patch") == sha256_bytes(result.p1_patch)


def test_role_specific_timeouts_keep_solver_and_repair_at_three_hours(tmp_path):
    result, roles, _, _ = run_controller(tmp_path, verdict="request_changes")

    assert result.state == WorkflowState.COMPLETED
    assert roles.timeouts == [
        (RoleName.SPECIFICATION_REVIEWER, 3_600),
        (RoleName.PATCH_REVIEWER, 3_600),
        (RoleName.REPAIR, 10_800),
    ]


def test_repair_out_of_scope_write_restores_final_and_falls_back_to_p0(tmp_path):
    class FinalWorktreeMutatingRunner(FakeRoleRunner):
        async def run_role(self, **kwargs):
            result = await super().run_role(**kwargs)
            if kwargs["role"] == RoleName.REPAIR:
                self.workspace.contents["final"] = b"out-of-scope write\n"
            return result

    workspace = FakeWorkspace()
    roles = FinalWorktreeMutatingRunner(workspace, verdict="request_changes")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
    )

    result = asyncio.run(controller.run(task(), existing_p0=existing_p0(tmp_path)))

    assert result.state == WorkflowState.PROTOCOL_VIOLATION
    assert result.repair_executed is True
    assert result.verdict == "request_changes"
    assert workspace.published == b"P0\n"
    assert workspace.contents["final"] == b""
    assert (artifacts.root / "outputs/discarded_final_worktree.patch").read_bytes() == (
        b"out-of-scope write\n"
    )


def test_initial_final_worktree_changes_are_recorded_then_normalized(tmp_path):
    workspace = FakeWorkspace(initial_final=b"image-provided change\n")

    result, _, workspace, artifacts = run_controller(tmp_path, workspace=workspace)

    assert result.state == WorkflowState.COMPLETED
    assert (artifacts.root / "inputs/environment.patch").read_bytes() == (
        b"image-provided change\n"
    )
    assert workspace.published == b"P0\n"


def test_existing_p0_accepts_safe_short_full_commit_equivalence(tmp_path):
    frozen = existing_p0(tmp_path)
    frozen = replace(frozen, base_commit="a" * 7)
    workspace = FakeWorkspace()
    roles = FakeRoleRunner(workspace)
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )

    result = asyncio.run(controller.run(task(), existing_p0=frozen))

    assert result.state == WorkflowState.COMPLETED
    assert git_commits_equivalent("A" * 7, "a" * 40)
    assert not git_commits_equivalent("a" * 6, "a" * 40)
    assert not git_commits_equivalent("not-a-commit", "a" * 40)


def test_json_correction_is_called_twice_then_stops(tmp_path):
    def configure(roles):
        roles.overrides[RoleName.SPECIFICATION_REVIEWER] = {"final_assistant_text": "not json"}
        roles.corrections = ["still not json", "also not json"]

    result, roles, workspace, artifacts = run_controller(tmp_path, configure=configure)
    assert result.state == WorkflowState.LEDGER_SCHEMA_FAILED
    assert roles.calls.count(RoleName.JSON_CORRECTOR) == 2
    assert RoleName.PATCH_REVIEWER not in roles.calls
    assert workspace.published == b"P0\n"
    validation = json.loads((artifacts.root / "reviewer/stage_a_validation.json").read_text())
    assert validation["normalization_used"] is False
    assert validation["correction_count"] == 2


def test_valid_single_correction_resumes_workflow(tmp_path):
    def configure(roles):
        roles.overrides[RoleName.SPECIFICATION_REVIEWER] = {"final_assistant_text": "not json"}
        roles.corrections = [json.dumps(valid_ledger())]

    result, roles, _, _ = run_controller(tmp_path, configure=configure)
    assert result.state == WorkflowState.COMPLETED
    assert roles.calls.count(RoleName.JSON_CORRECTOR) == 1


def test_fenced_reviewer_json_is_normalized_without_corrector(tmp_path):
    def configure(roles):
        roles.fence_roles = {
            RoleName.SPECIFICATION_REVIEWER,
            RoleName.PATCH_REVIEWER,
        }

    result, roles, _, artifacts = run_controller(tmp_path, configure=configure)
    assert result.state == WorkflowState.COMPLETED
    assert RoleName.JSON_CORRECTOR not in roles.calls
    for stage in ("stage_a", "stage_b"):
        validation = json.loads((artifacts.root / f"reviewer/{stage}_validation.json").read_text())
        assert validation["valid"] is True
        assert validation["normalization_used"] is True
        assert validation["normalization_method"] == "fenced_json"
        assert validation["normalization_methods"] == ["fenced_json"]
        assert validation["correction_count"] == 0


def test_embedded_json_after_correction_resumes_workflow(tmp_path):
    def configure(roles):
        roles.overrides[RoleName.SPECIFICATION_REVIEWER] = {"final_assistant_text": "not json"}
        roles.corrections = ["The corrected response follows:\n" + json.dumps(valid_ledger())]

    result, roles, _, artifacts = run_controller(tmp_path, configure=configure)
    assert result.state == WorkflowState.COMPLETED
    assert roles.calls.count(RoleName.JSON_CORRECTOR) == 1
    validation = json.loads((artifacts.root / "reviewer/stage_a_validation.json").read_text())
    assert validation["normalization_method"] == "embedded_json_object"
    assert validation["normalization_methods"] == ["embedded_json_object"]


def test_second_correction_can_fix_repository_evidence_error(tmp_path):
    corrected = valid_assessment(
        {
            "patch": {"sha256": sha256_bytes(b"P0\n")},
            "obligation_ledger": {"sha256": artifact_json_sha(valid_ledger())},
        }
    )

    def configure(roles):
        roles.overrides[RoleName.PATCH_REVIEWER] = {"final_assistant_text": "not json"}
        roles.corrections = [json.dumps(corrected), json.dumps(corrected)]

    workspace = FakeWorkspace(evidence_errors=[["O01 evidence path does not exist"], []])
    result, roles, _, artifacts = run_controller(
        tmp_path,
        configure=configure,
        workspace=workspace,
    )
    assert result.state == WorkflowState.COMPLETED
    assert roles.calls.count(RoleName.JSON_CORRECTOR) == 2
    validation = json.loads((artifacts.root / "reviewer/stage_b_validation.json").read_text())
    assert validation["valid"] is True
    assert validation["normalization_used"] is False
    assert validation["correction_count"] == 2


def test_protocol_digest_mismatch_is_not_sent_to_json_correction(tmp_path):
    def configure(roles):
        invalid_binding = valid_assessment(
            {
                "patch": {"sha256": "0" * 64},
                "obligation_ledger": {"sha256": "0" * 64},
            }
        )
        roles.overrides[RoleName.PATCH_REVIEWER] = {
            "final_assistant_text": json.dumps(invalid_binding)
        }

    result, roles, workspace, _ = run_controller(tmp_path, configure=configure)
    assert result.state == WorkflowState.PROTOCOL_VIOLATION
    assert RoleName.JSON_CORRECTOR not in roles.calls
    assert workspace.published == b"P0\n"


def test_v2_corrector_cannot_rewrite_parseable_source_semantics(tmp_path):
    class SourceRewritingRunner(FakeRoleRunner):
        async def run_role(self, **kwargs):
            result = await super().run_role(**kwargs)
            if kwargs["role"] == RoleName.SPECIFICATION_REVIEWER:
                invalid = json.loads(result.final_assistant_text)
                invalid["sources"][0]["locator"]["quote"] = "not present in the issue"
                corrected = json.loads(result.final_assistant_text)
                self.corrections = [json.dumps(corrected)]
                return replace(result, final_assistant_text=json.dumps(invalid))
            return result

    workspace = FakeWorkspace()
    roles = SourceRewritingRunner(workspace)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
        config=v2_config("E2"),
    )

    result = asyncio.run(controller.run(task(), existing_p0=existing_p0(tmp_path)))

    assert result.state == WorkflowState.PROTOCOL_VIOLATION
    assert "changed protected v2 semantic fields" in result.error
    assert roles.calls.count(RoleName.JSON_CORRECTOR) == 1
    assert RoleName.PATCH_REVIEWER not in roles.calls
    assert workspace.published == b"P0\n"
    source_validation = json.loads(
        (artifacts.root / "reviewer/source_anchor_validation.json").read_text()
    )
    assert source_validation["valid"] is False
    assert "not exact" in source_validation["errors"][0]["message"]


def test_v2_resume_rejects_stale_source_semantics_and_reruns_review(tmp_path):
    first, _, _, artifacts = run_controller(
        tmp_path,
        config=v2_config("E2", resume=True),
    )
    assert first.state == WorkflowState.COMPLETED

    ledger_path = artifacts.root / "reviewer/review_obligations.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["sources"][0]["locator"]["quote"] = "not present in the issue"
    ledger_path.write_text(json.dumps(ledger))
    validation_path = artifacts.root / "reviewer/stage_a_validation.json"
    validation = json.loads(validation_path.read_text())
    validation["payload_sha256"] = sha256_json(ledger)
    validation_path.write_text(json.dumps(validation))

    resumed, roles, _, artifacts = run_controller(
        tmp_path,
        config=v2_config("E2", resume=True),
    )

    assert resumed.state == WorkflowState.COMPLETED
    assert RoleName.SPECIFICATION_REVIEWER in roles.calls
    events = (artifacts.root / "events.jsonl").read_text()
    assert "LEDGER_RESUME_REJECTED" in events


def test_patch_reviewer_cannot_modify_frozen_contract_files(tmp_path):
    class ContractMutatingRunner(FakeRoleRunner):
        async def run_role(self, **kwargs):
            result = await super().run_role(**kwargs)
            if kwargs["role"] == RoleName.PATCH_REVIEWER:
                self.workspace.materialized["/contracts/review_obligations.json"] = b"{}"
            return result

    workspace = FakeWorkspace()
    roles = ContractMutatingRunner(workspace)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
    )
    result = asyncio.run(controller.run(task(), existing_p0=existing_p0(tmp_path)))

    assert result.state == WorkflowState.PROTOCOL_VIOLATION
    assert "modified frozen input" in result.error
    assert workspace.published == b"P0\n"


def test_stage_a_failure_never_enters_stage_b(tmp_path):
    def configure(roles):
        roles.overrides[RoleName.SPECIFICATION_REVIEWER] = {
            "exit_code": 2,
            "exception": "failed",
        }

    result, roles, _, _ = run_controller(tmp_path, configure=configure)
    assert result.state == WorkflowState.LEDGER_ROLE_FAILED
    assert RoleName.PATCH_REVIEWER not in roles.calls
    assert RoleName.REPAIR not in roles.calls


def test_reviewer_workspace_mutation_is_discarded_and_not_published(tmp_path):
    def configure(roles):
        roles.mutate_role = RoleName.SPECIFICATION_REVIEWER

    result, _, workspace, artifacts = run_controller(tmp_path, configure=configure)
    assert result.state == WorkflowState.PROTOCOL_VIOLATION
    assert "modified tracked" in result.error
    assert workspace.published == b"P0\n"
    assert artifacts.digest_file("outputs/p1.patch") == sha256_bytes(b"P0\n")


def test_v2_spec_reviewer_untracked_scratch_is_discarded_and_run_continues(tmp_path):
    class ScratchWritingRunner(FakeRoleRunner):
        async def run_role(self, **kwargs):
            result = await super().run_role(**kwargs)
            if kwargs["role"] == RoleName.SPECIFICATION_REVIEWER:
                self.workspace.untracked[kwargs["worktree"]] = b"scratch\n"
            return result

    workspace = FakeWorkspace()
    roles = ScratchWritingRunner(workspace)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
        config=v2_config("E2"),
    )

    result = asyncio.run(controller.run(task(), existing_p0=existing_p0(tmp_path)))

    assert result.state == WorkflowState.COMPLETED
    assert workspace.untracked["spec"] == b""
    discarded = artifacts.root / "reviewer/stage_a_discarded_untracked.patch"
    assert discarded.read_bytes() == b"scratch\n"
    manifest = json.loads((artifacts.root / "manifest.json").read_text())
    assert manifest["spec_reviewer_discarded_untracked_sha256"] == sha256_bytes(b"scratch\n")
    assert (
        "SPECIFICATION_REVIEWER_UNTRACKED_DISCARDED"
        in (artifacts.root / "events.jsonl").read_text()
    )


def test_v2_spec_reviewer_tracked_mutation_remains_protocol_violation(tmp_path):
    workspace = FakeWorkspace()
    roles = FakeRoleRunner(workspace)
    roles.mutate_role = RoleName.SPECIFICATION_REVIEWER
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
        config=v2_config("E2"),
    )

    result = asyncio.run(controller.run(task(), existing_p0=existing_p0(tmp_path)))

    assert result.state == WorkflowState.PROTOCOL_VIOLATION
    assert "modified base-tracked" in result.error
    assert workspace.published == b"P0\n"


def test_v2_patch_reviewer_untracked_scratch_is_discarded_and_run_continues(tmp_path):
    class ScratchWritingRunner(FakeRoleRunner):
        async def run_role(self, **kwargs):
            result = await super().run_role(**kwargs)
            if kwargs["role"] == RoleName.PATCH_REVIEWER:
                self.workspace.untracked[kwargs["worktree"]] = b"scratch\n"
            return result

    workspace = FakeWorkspace()
    roles = ScratchWritingRunner(workspace)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
        config=v2_config("E2"),
    )

    result = asyncio.run(controller.run(task(), existing_p0=existing_p0(tmp_path)))

    assert result.state == WorkflowState.COMPLETED
    assert workspace.untracked["patch"] == b""
    assert workspace.contents["patch"] == workspace.p0


def test_v2_patch_reviewer_tracked_mutation_remains_protocol_violation(tmp_path):
    class TrackedWritingRunner(FakeRoleRunner):
        async def run_role(self, **kwargs):
            result = await super().run_role(**kwargs)
            if kwargs["role"] == RoleName.PATCH_REVIEWER:
                self.workspace.contents[kwargs["worktree"]] += b"mutation\n"
            return result

    workspace = FakeWorkspace()
    roles = TrackedWritingRunner(workspace)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
        config=v2_config("E2"),
    )

    result = asyncio.run(controller.run(task(), existing_p0=existing_p0(tmp_path)))

    assert result.state == WorkflowState.PROTOCOL_VIOLATION
    assert "modified base-tracked" in result.error
    assert workspace.published == b"P0\n"


def test_prompt_json_escaping_and_hashes_are_stable(tmp_path):
    result, _, _, artifacts = run_controller(tmp_path, config=v2_config("E2"))
    prompt = (artifacts.root / "prompts/build_obligations.rendered.md").read_text()
    assert '\\"quotes\\"' in prompt
    assert '"title": "ReviewObligationsV2"' in prompt
    assert "{{REVIEW_OBLIGATIONS_OUTPUT_SCHEMA}}" not in prompt
    assessment_prompt = (artifacts.root / "prompts/assess_patch.rendered.md").read_text()
    assert '"title": "PatchAssessmentV2"' in assessment_prompt
    assert '"test_name"' in assessment_prompt
    assert "{{PATCH_ASSESSMENT_OUTPUT_SCHEMA}}" not in assessment_prompt
    first = artifacts.digest_file("prompts/build_obligations.rendered.md")
    assert first == sha256_bytes(prompt.encode())
    assert result.p0_sha256 == sha256_bytes(b"P0\n")


def test_resume_does_not_repeat_validated_review_stages(tmp_path):
    first, _, _, _ = run_controller(tmp_path)
    assert first.state == WorkflowState.COMPLETED
    resumed, roles, _, artifacts = run_controller(
        tmp_path,
        config=WorkflowConfig(resume=True),
    )
    assert resumed.state == WorkflowState.COMPLETED
    assert RoleName.SPECIFICATION_REVIEWER not in roles.calls
    assert RoleName.PATCH_REVIEWER not in roles.calls
    assert not (artifacts.root / "reviewer/source_anchor_validation.json").exists()


def test_artifact_store_redacts_secrets_in_events_and_text(tmp_path):
    store = ArtifactStore(tmp_path, secrets=("secret-token",))
    store.write_text("output.txt", "value=secret-token")
    store.event("TEST", value="secret-token")
    assert "secret-token" not in (tmp_path / "output.txt").read_text()
    assert "secret-token" not in (tmp_path / "events.jsonl").read_text()
    with pytest.raises(ValueError, match="unredacted secret"):
        store.write_bytes("raw.bin", b"secret-token")


def v2_config(arm, *, resume=False):
    return WorkflowConfig(
        contract_version="2.0",
        experiment_arm=arm,
        prompt_version="2.2",
        controller_version="2.0.0",
        resume=resume,
    )


def test_experiment_arms_are_versioned_and_reversible():
    WorkflowConfig().validate()
    v2_config("E1").validate()
    v2_config("E2").validate()
    WorkflowConfig(
        contract_version="3.0",
        experiment_arm="E1",
        max_obligations=64,
    ).validate()
    WorkflowConfig(
        contract_version="3.0",
        experiment_arm="E2",
        max_obligations=64,
    ).validate()
    WorkflowConfig(
        contract_version="3.0",
        experiment_arm="E2",
        max_obligations=64,
        prompt_version="4.0",
    ).validate()
    with pytest.raises(ValueError, match="requires contract"):
        WorkflowConfig(contract_version="2.0", experiment_arm="E0").validate()
    with pytest.raises(ValueError, match="requires contract"):
        WorkflowConfig(contract_version="1.0", experiment_arm="E2").validate()
    with pytest.raises(ValueError, match="source policy version"):
        WorkflowConfig(
            contract_version="2.0",
            experiment_arm="E2",
            source_policy_version="2.0",
        ).validate()
    with pytest.raises(ValueError, match="prompt v4.0 requires"):
        WorkflowConfig(
            contract_version="2.0",
            experiment_arm="E2",
            prompt_version="4.0",
        ).validate()


def test_v2_e1_uses_source_contracts_without_repair_audit_gate(tmp_path):
    result, roles, workspace, artifacts = run_controller(
        tmp_path,
        verdict="request_changes",
        config=v2_config("E1"),
    )

    assert result.state == WorkflowState.COMPLETED
    assert result.repair_executed is True
    assert workspace.published == b"P1\n"
    assert roles.calls.count(RoleName.REPAIR) == 1
    assert (artifacts.root / "inputs/source_policy.json").is_file()
    assert (artifacts.root / "reviewer/source_anchor_validation.json").is_file()
    assert (artifacts.root / "reviewer/effective_review_set.json").is_file()
    assert not (artifacts.root / "repair/repair_audit.json").exists()
    manifest = json.loads((artifacts.root / "manifest.json").read_text())
    assert manifest["experiment_arm"] == "E1"
    assert manifest["contract_version"] == "2.0"
    assert manifest["repair_audit_status"] == "not_required"


def test_v2_e2_validates_repair_audit_before_publishing_candidate(tmp_path):
    result, roles, workspace, artifacts = run_controller(
        tmp_path,
        verdict="request_changes",
        config=v2_config("E2"),
    )

    assert result.state == WorkflowState.COMPLETED
    assert result.repair_audit is not None
    assert workspace.published == b"P1\n"
    assert roles.calls.count(RoleName.REPAIR) == 1
    validation = json.loads((artifacts.root / "repair/repair_audit_validation.json").read_text())
    assert validation["valid"] is True
    sequence = json.loads((artifacts.root / "repair/repair_sequence_observation.json").read_text())
    assert sequence["status"] == "unavailable"
    manifest = json.loads((artifacts.root / "manifest.json").read_text())
    assert manifest["repair_audit_status"] == "validated"
    assert manifest["effective_review_set_sha256"]
    assert manifest["repair_sequence_observation_status"] == "unavailable"
    assert manifest["repair_sequence_observation_sha256"]
    assert "external_agents_eval/agents/pier/pi_role_runner.py" in manifest["implementation_sha256"]
    assert (
        "external_agents_eval/workflows/solve_review_repair/observability.py"
        in manifest["implementation_sha256"]
    )


def test_v2_e2_missing_repair_audit_keeps_diagnostic_diff_and_falls_back(tmp_path):
    def configure(roles):
        roles.write_repair_audit = False

    result, roles, workspace, artifacts = run_controller(
        tmp_path,
        verdict="request_changes",
        configure=configure,
        config=v2_config("E2"),
    )

    assert result.state == WorkflowState.REPAIR_AUDIT_FAILED
    assert roles.calls.count(RoleName.REPAIR) == 1
    assert workspace.published == b"P0\n"
    assert (artifacts.root / "repair/discarded_candidate.patch").read_bytes() == b"P1\n"
    validation = json.loads((artifacts.root / "repair/repair_audit_validation.json").read_text())
    assert validation["valid"] is False
    manifest = json.loads((artifacts.root / "manifest.json").read_text())
    assert manifest["repair_audit_status"] == "failed"


def test_v2_abstain_skips_repair_but_is_not_reported_as_approve(tmp_path):
    result, roles, workspace, artifacts = run_controller(
        tmp_path,
        verdict="abstain",
        config=v2_config("E2"),
    )

    assert result.state == WorkflowState.COMPLETED
    assert result.verdict == "abstain"
    assert result.repair_executed is False
    assert RoleName.REPAIR not in roles.calls
    assert workspace.published == b"P0\n"
    manifest = json.loads((artifacts.root / "manifest.json").read_text())
    assert manifest["abstain_reason"] == "The required runtime is unavailable."
    assert manifest["repair_audit_status"] == "not_applicable"


def test_v2_e2_repair_timeout_keeps_partial_patch_and_marks_audit_incomplete(tmp_path):
    def configure(roles):
        roles.overrides[RoleName.REPAIR] = {
            "exit_code": 124,
            "timed_out": True,
            "exception": "timeout",
        }

    result, _, workspace, artifacts = run_controller(
        tmp_path,
        verdict="request_changes",
        configure=configure,
        config=v2_config("E2"),
    )

    assert result.state == WorkflowState.REPAIR_FAILED
    assert workspace.published == b"P1\n"
    validation = json.loads((artifacts.root / "repair/repair_audit_validation.json").read_text())
    assert validation == {
        "errors": [
            {
                "kind": "timeout",
                "message": "Repair did not finish; audit completeness was not asserted",
            }
        ],
        "incomplete": True,
        "schema_version": "1.0",
        "valid": False,
    }
    manifest = json.loads((artifacts.root / "manifest.json").read_text())
    assert manifest["repair_audit_status"] == "incomplete"
