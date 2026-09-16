from __future__ import annotations

import re
from typing import Any

from .artifacts import ArtifactStore, sha256_bytes, sha256_json
from .config import WorkflowConfig
from .contracts_v2 import changed_paths_between_patches, patch_changed_paths
from .models import (
    ExistingPatch,
    RoleName,
    RoleRunner,
    RoleRunResult,
    TaskInput,
    WorkflowResult,
    WorkflowState,
    WorkspaceManager,
)
from .observability import observe_repair_sequence
from .prompt_renderer import PromptRenderer
from .validators import (
    ContractValidationError,
    ContractValidator,
    extract_single_json_object,
)

EMPTY_DIFF_SHA256 = sha256_bytes(b"")


def _project_contract_array(value: Any, item_projector) -> Any:
    if not isinstance(value, list):
        return value
    return [item_projector(item) for item in value]


def _project_contract_object(
    value: Any,
    fields: tuple[str, ...],
    *,
    child_projectors: dict[str, Any] | None = None,
    neutral_defaults: dict[str, Any] | None = None,
) -> Any:
    """Project recognized fields while retaining non-neutral missing-field state."""
    if not isinstance(value, dict):
        return value
    child_projectors = child_projectors or {}
    neutral_defaults = neutral_defaults or {}
    projection = []
    for field in fields:
        if field in value:
            present = True
            field_value = value[field]
        elif field in neutral_defaults:
            present = True
            field_value = neutral_defaults[field]
        else:
            present = False
            field_value = None
        if present and field in child_projectors:
            field_value = child_projectors[field](field_value)
        projection.append(
            {
                "field": field,
                "present": present,
                "value": field_value,
            }
        )
    return projection


def _project_v3_source(value: Any) -> Any:
    common = ("id", "kind")
    if isinstance(value, dict) and value.get("kind") == "task":
        fields = (*common, "field", "quote")
    elif isinstance(value, dict) and value.get("kind") == "repository":
        fields = (*common, "path", "line_start", "line_end")
    else:
        # Preserve every recognized source field when the discriminator itself is invalid.
        fields = (*common, "field", "quote", "path", "line_start", "line_end")
    return _project_contract_object(value, fields)


def _project_v3_obligation(value: Any) -> Any:
    return _project_contract_object(
        value,
        ("id", "statement", "criticality", "basis", "source_refs", "rationale"),
    )


def _project_v3_open_question(value: Any) -> Any:
    return _project_contract_object(value, ("id", "question", "source_refs"))


def _project_v3_implementation_evidence(value: Any) -> Any:
    return _project_contract_object(
        value,
        ("path", "line_start", "line_end", "explanation"),
    )


def _project_v3_validation_evidence(value: Any) -> Any:
    return _project_contract_object(value, ("command", "result", "explanation"))


def _project_v3_assessment_item(value: Any) -> Any:
    neutral_defaults: dict[str, Any] = {
        "implementation_evidence": [],
        "validation_evidence": [],
    }
    if isinstance(value, dict):
        if value.get("source_status") == "supported":
            neutral_defaults["source_note"] = None
        if value.get("implementation_status") == "covered" or value.get(
            "source_status"
        ) in {"unsupported", "unclear"}:
            neutral_defaults["gap"] = None
    return _project_contract_object(
        value,
        (
            "obligation_ref",
            "source_status",
            "source_note",
            "implementation_status",
            "implementation_evidence",
            "validation_evidence",
            "gap",
        ),
        child_projectors={
            "implementation_evidence": lambda items: _project_contract_array(
                items, _project_v3_implementation_evidence
            ),
            "validation_evidence": lambda items: _project_contract_array(
                items, _project_v3_validation_evidence
            ),
        },
        neutral_defaults=neutral_defaults,
    )


def _project_v3_regression(value: Any) -> Any:
    return _project_contract_object(
        value,
        (
            "id",
            "statement",
            "source_refs",
            "blocking",
            "implementation_evidence",
            "validation_evidence",
        ),
        child_projectors={
            "implementation_evidence": lambda items: _project_contract_array(
                items, _project_v3_implementation_evidence
            ),
            "validation_evidence": lambda items: _project_contract_array(
                items, _project_v3_validation_evidence
            ),
        },
        neutral_defaults={
            "implementation_evidence": [],
            "validation_evidence": [],
        },
    )


def _project_v3_repair_request(value: Any) -> Any:
    return _project_contract_object(
        value,
        (
            "id",
            "finding_refs",
            "action",
            "problem",
            "required_outcome",
            "preserve",
            "validation",
        ),
        neutral_defaults={"preserve": []},
    )


def _project_v3_correction_semantics(schema_name: str, payload: dict[str, Any]) -> Any:
    if schema_name == "ledger":
        return _project_contract_object(
            payload,
            (
                "schema_version",
                "task_id",
                "summary",
                "sources",
                "obligations",
                "open_questions",
            ),
            child_projectors={
                "sources": lambda items: _project_contract_array(items, _project_v3_source),
                "obligations": lambda items: _project_contract_array(
                    items, _project_v3_obligation
                ),
                "open_questions": lambda items: _project_contract_array(
                    items, _project_v3_open_question
                ),
            },
            neutral_defaults={"open_questions": []},
        )
    if schema_name == "assessment":
        return _project_contract_object(
            payload,
            (
                "schema_version",
                "task_id",
                "patch_sha256",
                "ledger_sha256",
                "additional_sources",
                "late_obligations",
                "obligation_assessments",
                "unexpected_regressions",
                "repair_requests",
                "verdict",
                "verdict_rationale",
            ),
            child_projectors={
                "additional_sources": lambda items: _project_contract_array(
                    items, _project_v3_source
                ),
                "late_obligations": lambda items: _project_contract_array(
                    items, _project_v3_obligation
                ),
                "obligation_assessments": lambda items: _project_contract_array(
                    items, _project_v3_assessment_item
                ),
                "unexpected_regressions": lambda items: _project_contract_array(
                    items, _project_v3_regression
                ),
                "repair_requests": lambda items: _project_contract_array(
                    items, _project_v3_repair_request
                ),
            },
            neutral_defaults={
                "additional_sources": [],
                "late_obligations": [],
                "unexpected_regressions": [],
                "repair_requests": [],
            },
        )
    return None


class ProtocolViolationError(RuntimeError):
    pass


def git_commits_equivalent(left: str, right: str) -> bool:
    """Accept a safe abbreviated/full Git object-id pair in either order."""
    object_id = re.compile(r"[0-9a-fA-F]{7,40}")
    if object_id.fullmatch(left) is None or object_id.fullmatch(right) is None:
        return False
    left = left.lower()
    right = right.lower()
    return left.startswith(right) or right.startswith(left)


class SolveReviewRepairController:
    """Drive one solver/replay, two isolated reviews, and at most one repair."""

    def __init__(
        self,
        *,
        role_runner: RoleRunner,
        workspace: WorkspaceManager,
        artifacts: ArtifactStore,
        config: WorkflowConfig | None = None,
    ) -> None:
        self.role_runner = role_runner
        self.workspace = workspace
        self.artifacts = artifacts
        self.config = config or WorkflowConfig()
        self.config.validate()
        self.validator = ContractValidator(
            self.config.schema_dir,
            contract_version=self.config.contract_version,
        )
        self.renderer = PromptRenderer(self.config.prompt_dir)
        self._role_results: list[tuple[RoleName, RoleRunResult]] = []
        self._prompt_hashes: dict[str, str] = {}
        self._existing_p0: ExistingPatch | None = None
        self._source_policy_sha256: str | None = None
        self._source_bindings_sha256: str | None = None
        self._effective_review_set_sha256: str | None = None
        self._repair_audit_status = "not_applicable"
        self._repair_sequence_observation_status = "not_applicable"
        self._repair_sequence_observation_sha256: str | None = None
        self._unresolved_review_items: list[str] = []
        self._abstain_reason: str | None = None
        self._spec_reviewer_discarded_patch_sha256: str | None = None

    async def run(
        self,
        task: TaskInput,
        *,
        existing_p0: ExistingPatch | None = None,
    ) -> WorkflowResult:
        self._role_results = []
        self._prompt_hashes = {}
        state = WorkflowState.INITIALIZED
        p0 = b""
        p0_sha = sha256_bytes(p0)
        ledger: dict[str, Any] | None = None
        assessment: dict[str, Any] | None = None
        effective_review_set: dict[str, Any] | None = None
        repair_audit: dict[str, Any] | None = None
        verdict: str | None = None
        repair_executed = False
        repair_error: str | None = None
        self.artifacts.event(state.value, task_id=task.task_id)
        self._existing_p0 = existing_p0
        self._source_policy_sha256 = None
        self._source_bindings_sha256 = None
        self._effective_review_set_sha256 = None
        self._repair_audit_status = "not_applicable"
        self._repair_sequence_observation_status = "not_applicable"
        self._repair_sequence_observation_sha256 = None
        self._unresolved_review_items = []
        self._abstain_reason = None
        self._spec_reviewer_discarded_patch_sha256 = None
        worktrees = await self.workspace.prepare(base_commit=task.base_commit)
        try:
            if existing_p0 is not None:
                try:
                    p0 = self._load_existing_p0(task, existing_p0)
                    p0_sha = sha256_bytes(p0)
                    state = WorkflowState.P0_LOADED
                except Exception as error:
                    state = WorkflowState.P0_LOAD_FAILED
                    self.artifacts.event(state.value, error=str(error))
                    self._write_run_metadata_artifacts()
                    self._write_manifest(
                        task=task,
                        state=state,
                        p0_sha=p0_sha,
                        p1_sha=p0_sha,
                        verdict=None,
                        repair_executed=False,
                        existing_p0=existing_p0,
                        error=str(error),
                    )
                    raise
            else:
                try:
                    solve_result = await self._run_role(
                        role=RoleName.SOLVER,
                        prompt=task.issue_body,
                        worktree=worktrees.solver,
                        prefix="solver",
                    )
                except Exception as error:
                    state = WorkflowState.P0_LOAD_FAILED
                    self.artifacts.event(state.value, error=str(error))
                    self._write_run_metadata_artifacts()
                    self._write_manifest(
                        task=task,
                        state=state,
                        p0_sha=p0_sha,
                        p1_sha=p0_sha,
                        verdict=None,
                        repair_executed=False,
                        existing_p0=None,
                        error=str(error),
                    )
                    raise
                try:
                    self._require_role_success(solve_result, "Solver")
                except RuntimeError as error:
                    try:
                        p0 = await self.workspace.capture_patch(
                            worktree=worktrees.solver,
                            base_commit=task.base_commit,
                        )
                    except Exception as capture_error:  # noqa: BLE001 - preserve verifier path
                        p0 = b""
                        self.artifacts.event(
                            "SOLVER_PARTIAL_PATCH_CAPTURE_FAILED",
                            error=str(capture_error),
                        )
                    p0_sha = sha256_bytes(p0)
                    self.artifacts.write_bytes("inputs/p0.patch", p0)
                    self.artifacts.event(
                        WorkflowState.SOLVER_FAILED.value,
                        error=str(error),
                        partial_patch_captured=bool(p0),
                        p0_sha256=p0_sha,
                    )
                    await self.workspace.discard_worktree(worktree=worktrees.solver)
                    return await self._fallback(
                        state=WorkflowState.SOLVER_FAILED,
                        task=task,
                        p0=p0,
                        p0_sha=p0_sha,
                        error=error,
                    )
                try:
                    p0 = await self.workspace.capture_patch(
                        worktree=worktrees.solver,
                        base_commit=task.base_commit,
                    )
                    p0_sha = sha256_bytes(p0)
                    state = WorkflowState.P0_SOLVED
                except Exception as error:
                    state = WorkflowState.P0_LOAD_FAILED
                    self.artifacts.event(state.value, error=str(error))
                    self._write_run_metadata_artifacts()
                    self._write_manifest(
                        task=task,
                        state=state,
                        p0_sha=p0_sha,
                        p1_sha=p0_sha,
                        verdict=None,
                        repair_executed=False,
                        existing_p0=None,
                        error=str(error),
                    )
                    raise
            self.artifacts.write_bytes("inputs/p0.patch", p0)
            self.artifacts.event(state.value, p0_sha256=p0_sha)
            await self.workspace.discard_worktree(worktree=worktrees.solver)
            await self._restore_final_worktree(
                base_commit=task.base_commit,
                artifact_path="inputs/environment.patch",
                event="FINAL_WORKTREE_NORMALIZED",
            )

            source_policy_path: str | None = None
            if self.config.contract_version == "2.0":
                source_policy = self._source_policy(task)
                self.validator.validate_source_policy(source_policy, task_id=task.task_id)
                self.artifacts.write_json("inputs/source_policy.json", source_policy)
                self._source_policy_sha256 = self.artifacts.digest_file("inputs/source_policy.json")
                source_policy_path = await self.workspace.materialize_file(
                    name="source_policy.json",
                    payload=(self.artifacts.root / "inputs/source_policy.json").read_bytes(),
                    readonly=True,
                )

            state = WorkflowState.LEDGER_RUNNING
            self.artifacts.event(state.value)
            build_input = self._build_obligations_input(
                task,
                worktrees.spec_review,
                source_policy_path=source_policy_path,
            )
            self.validator.validate_schema("build_input", build_input)
            self.artifacts.write_json("inputs/build_obligations_input.json", build_input)
            stage_a_prompt, stage_a_hash = self.renderer.render(
                self.config.prompt_templates["build_obligations"],
                {
                    "STAGE_A_FEW_SHOT_EXAMPLES": self.renderer.load_few_shots(
                        self.config.stage_a_few_shots
                    ),
                    "REVIEW_OBLIGATIONS_OUTPUT_SCHEMA": self.renderer.json_text(
                        self.validator.schema("ledger")
                    ),
                    "BUILD_OBLIGATIONS_INPUT_JSON": self.renderer.json_text(build_input),
                },
            )
            self.artifacts.write_text("prompts/build_obligations.rendered.md", stage_a_prompt)
            self._prompt_hashes["build_obligations"] = stage_a_hash
            ledger = self._resume_json(
                "reviewer/stage_a_validation.json", "reviewer/review_obligations.json"
            )
            if ledger is not None:
                try:
                    self.validator.validate_ledger(
                        ledger,
                        task_id=task.task_id,
                        source_policy_sha256=self._source_policy_sha256,
                    )
                    if self.config.contract_version in {"2.0", "3.0"}:
                        source_errors = await self._record_ledger_source_validation(
                            task=task,
                            worktree=worktrees.spec_review,
                            sources=ledger.get("sources", []),
                        )
                        if source_errors:
                            raise ContractValidationError("source", tuple(source_errors))
                except ContractValidationError as error:
                    self.artifacts.event(
                        "LEDGER_RESUME_REJECTED",
                        errors=error.as_json(),
                    )
                    ledger = None
                else:
                    self.artifacts.event("LEDGER_RESUMED")
            if ledger is None:
                try:
                    stage_a_result = await self._run_role(
                        role=RoleName.SPECIFICATION_REVIEWER,
                        prompt=stage_a_prompt,
                        worktree=worktrees.spec_review,
                        prefix="reviewer/stage_a",
                    )
                    if self.config.contract_version in {"2.0", "3.0"}:
                        await self._require_reviewer_tracked_files_unchanged(
                            worktrees.spec_review,
                            task.base_commit,
                            "Specification Reviewer",
                        )
                        await self._restore_spec_reviewer_worktree(
                            worktree=worktrees.spec_review,
                            base_commit=task.base_commit,
                        )
                    else:
                        await self._require_reviewer_unchanged(
                            worktrees.spec_review,
                            task.base_commit,
                            "Specification Reviewer",
                        )
                    await self._require_pristine_worktrees(
                        worktrees=(
                            worktrees.patch_review,
                            worktrees.repair,
                            worktrees.final,
                        ),
                        base_commit=task.base_commit,
                        role="Specification Reviewer",
                    )
                    if source_policy_path is not None:
                        await self._require_materialized_file_unchanged(
                            path=source_policy_path,
                            expected=self._source_policy_sha256,
                            role="Specification Reviewer",
                        )
                    self._require_role_success(stage_a_result, "Specification Reviewer")
                except ProtocolViolationError as error:
                    return await self._fallback(
                        state=WorkflowState.PROTOCOL_VIOLATION,
                        task=task,
                        p0=p0,
                        p0_sha=p0_sha,
                        error=error,
                    )
                except Exception as error:  # noqa: BLE001 - role failures become frozen P0
                    return await self._fallback(
                        state=WorkflowState.LEDGER_ROLE_FAILED,
                        task=task,
                        p0=p0,
                        p0_sha=p0_sha,
                        error=error,
                    )
                self.artifacts.write_text(
                    "reviewer/review_obligations.raw.txt", stage_a_result.final_assistant_text
                )
                try:
                    ledger = await self._validate_with_corrections(
                        raw=stage_a_result.final_assistant_text,
                        schema_name="ledger",
                        semantic_validate=lambda payload: self.validator.validate_ledger(
                            payload,
                            task_id=task.task_id,
                            source_policy_sha256=self._source_policy_sha256,
                        ),
                        worktree=worktrees.correction,
                        prefix="reviewer/stage_a_correction",
                        async_validate=(
                            (
                                lambda payload: self._record_ledger_source_validation(
                                    task=task,
                                    worktree=worktrees.spec_review,
                                    sources=payload.get("sources", []),
                                )
                            )
                            if self.config.contract_version in {"2.0", "3.0"}
                            else None
                        ),
                    )
                except ContractValidationError as error:
                    if error.kind == "protocol":
                        failure_state = WorkflowState.PROTOCOL_VIOLATION
                    else:
                        failure_state = (
                            WorkflowState.LEDGER_SCHEMA_FAILED
                            if error.kind == "schema"
                            else WorkflowState.LEDGER_SEMANTIC_FAILED
                        )
                    return await self._fallback(
                        state=failure_state,
                        task=task,
                        p0=p0,
                        p0_sha=p0_sha,
                        error=error,
                    )
                self.artifacts.write_json("reviewer/review_obligations.json", ledger)
            if self.config.contract_version in {"2.0", "3.0"}:
                state = WorkflowState.LEDGER_SOURCE_ANCHORS_VALIDATED
                self.artifacts.event(state.value)
            ledger_sha = self.artifacts.digest_file("reviewer/review_obligations.json")
            await self.workspace.apply_patch(worktree=worktrees.patch_review, patch=p0)
            applied_p0_digest = await self.workspace.tracked_diff_digest(
                worktree=worktrees.patch_review,
                base_commit=task.base_commit,
            )
            expected_p0_digest = (
                existing_p0.tracked_diff_sha256 if existing_p0 is not None else p0_sha
            )
            if applied_p0_digest != expected_p0_digest:
                return await self._fallback(
                    state=WorkflowState.P0_LOAD_FAILED,
                    task=task,
                    p0=p0,
                    p0_sha=p0_sha,
                    error=ValueError(
                        "Applied P0 tracked diff digest does not match the source manifest"
                    ),
                    ledger=ledger,
                )
            expected_p0_tracked_digest = await self.workspace.tracked_files_diff_digest(
                worktree=worktrees.patch_review,
                base_commit=task.base_commit,
            )
            p0_container_path = await self.workspace.materialize_file(
                name="p0.patch", payload=p0, readonly=True
            )
            ledger_container_path = await self.workspace.materialize_file(
                name="review_obligations.json",
                payload=(self.artifacts.root / "reviewer/review_obligations.json").read_bytes(),
                readonly=True,
            )
            source_bindings_container_path: str | None = None
            if self.config.contract_version == "3.0":
                assert self._source_bindings_sha256 is not None
                source_bindings_container_path = await self.workspace.materialize_file(
                    name="source_bindings.json",
                    payload=(
                        self.artifacts.root / "reviewer/source_bindings.json"
                    ).read_bytes(),
                    readonly=True,
                )
            state = WorkflowState.LEDGER_VALIDATED
            self.artifacts.event(state.value, ledger_sha256=ledger_sha)

            state = WorkflowState.ASSESSMENT_RUNNING
            self.artifacts.event(state.value)
            assess_input = self._assess_patch_input(
                task,
                worktrees.patch_review,
                p0_sha,
                ledger_sha,
                p0_container_path,
                ledger_container_path,
                source_policy_path=source_policy_path,
                source_bindings_path=source_bindings_container_path,
            )
            self.validator.validate_schema("assessment_input", assess_input)
            self.artifacts.write_json("inputs/assess_patch_input.json", assess_input)
            stage_b_prompt, stage_b_hash = self.renderer.render(
                self.config.prompt_templates["assess_patch"],
                {
                    "STAGE_B_FEW_SHOT_EXAMPLES": self.renderer.load_few_shots(
                        self.config.stage_b_few_shots
                    ),
                    "PATCH_ASSESSMENT_OUTPUT_SCHEMA": self.renderer.json_text(
                        self.validator.schema("assessment")
                    ),
                    "ASSESS_PATCH_INPUT_JSON": self.renderer.json_text(assess_input),
                },
            )
            self.artifacts.write_text("prompts/assess_patch.rendered.md", stage_b_prompt)
            self._prompt_hashes["assess_patch"] = stage_b_hash
            assessment = self._resume_json(
                "reviewer/stage_b_validation.json", "reviewer/patch_assessment.json"
            )
            if assessment is not None:
                try:
                    self.validator.validate_assessment(
                        assessment,
                        task_id=task.task_id,
                        patch_sha256=p0_sha,
                        ledger_sha256=ledger_sha,
                        ledger=ledger,
                    )
                    combined_errors = await self._assessment_validation_errors(
                        task=task,
                        evidence_worktree=worktrees.patch_review,
                        source_worktree=worktrees.spec_review,
                        ledger=ledger,
                        assessment=assessment,
                    )
                    if combined_errors:
                        raise ContractValidationError("semantic", tuple(combined_errors))
                except ContractValidationError as error:
                    self.artifacts.event(
                        "ASSESSMENT_RESUME_REJECTED",
                        errors=error.as_json(),
                    )
                    assessment = None
                else:
                    self.artifacts.event("ASSESSMENT_RESUMED")
            if assessment is None:
                try:
                    stage_b_result = await self._run_role(
                        role=RoleName.PATCH_REVIEWER,
                        prompt=stage_b_prompt,
                        worktree=worktrees.patch_review,
                        prefix="reviewer/stage_b",
                    )
                    await self._require_patch_reviewer_frozen_p0(
                        worktree=worktrees.patch_review,
                        base_commit=task.base_commit,
                        expected_tracked_digest=expected_p0_tracked_digest,
                        p0=p0,
                        role="Patch Reviewer",
                    )
                    await self._require_pristine_worktrees(
                        worktrees=(
                            worktrees.spec_review,
                            worktrees.repair,
                            worktrees.final,
                        ),
                        base_commit=task.base_commit,
                        role="Patch Reviewer",
                    )
                    await self._require_materialized_file_unchanged(
                        path=p0_container_path,
                        expected=p0_sha,
                        role="Patch Reviewer",
                    )
                    await self._require_materialized_file_unchanged(
                        path=ledger_container_path,
                        expected=ledger_sha,
                        role="Patch Reviewer",
                    )
                    if source_policy_path is not None:
                        await self._require_materialized_file_unchanged(
                            path=source_policy_path,
                            expected=self._source_policy_sha256,
                            role="Patch Reviewer",
                        )
                    if source_bindings_container_path is not None:
                        await self._require_materialized_file_unchanged(
                            path=source_bindings_container_path,
                            expected=self._source_bindings_sha256,
                            role="Patch Reviewer",
                        )
                    await self._restore_patch_reviewer_worktree(
                        worktree=worktrees.patch_review,
                        base_commit=task.base_commit,
                        p0=p0,
                        expected_p0_digest=expected_p0_digest,
                    )
                    self._require_role_success(stage_b_result, "Patch Reviewer")
                except ProtocolViolationError as error:
                    return await self._fallback(
                        state=WorkflowState.PROTOCOL_VIOLATION,
                        task=task,
                        p0=p0,
                        p0_sha=p0_sha,
                        error=error,
                        ledger=ledger,
                    )
                except Exception as error:  # noqa: BLE001 - role failures become frozen P0
                    return await self._fallback(
                        state=WorkflowState.ASSESSMENT_ROLE_FAILED,
                        task=task,
                        p0=p0,
                        p0_sha=p0_sha,
                        error=error,
                        ledger=ledger,
                    )
                self.artifacts.write_text(
                    "reviewer/patch_assessment.raw.txt", stage_b_result.final_assistant_text
                )
                try:
                    assessment = await self._validate_with_corrections(
                        raw=stage_b_result.final_assistant_text,
                        schema_name="assessment",
                        semantic_validate=lambda payload: self.validator.validate_assessment(
                            payload,
                            task_id=task.task_id,
                            patch_sha256=p0_sha,
                            ledger_sha256=ledger_sha,
                            ledger=ledger,
                        ),
                        worktree=worktrees.patch_review,
                        prefix="reviewer/stage_b_correction",
                        async_validate=lambda payload: self._assessment_validation_errors(
                            task=task,
                            evidence_worktree=worktrees.patch_review,
                            source_worktree=worktrees.spec_review,
                            ledger=ledger,
                            assessment=payload,
                        ),
                    )
                except ContractValidationError as error:
                    if error.kind == "protocol":
                        failure_state = WorkflowState.PROTOCOL_VIOLATION
                    else:
                        failure_state = (
                            WorkflowState.ASSESSMENT_SCHEMA_FAILED
                            if error.kind == "schema"
                            else WorkflowState.ASSESSMENT_SEMANTIC_FAILED
                        )
                    return await self._fallback(
                        state=failure_state,
                        task=task,
                        p0=p0,
                        p0_sha=p0_sha,
                        error=error,
                        ledger=ledger,
                    )
                self.artifacts.write_json("reviewer/patch_assessment.json", assessment)
            if self.config.contract_version in {"2.0", "3.0"}:
                source_errors = await self._record_assessment_source_validation(
                    task=task,
                    worktree=worktrees.spec_review,
                    ledger=ledger,
                    assessment=assessment,
                )
                if source_errors:
                    return await self._fallback(
                        state=WorkflowState.ASSESSMENT_SEMANTIC_FAILED,
                        task=task,
                        p0=p0,
                        p0_sha=p0_sha,
                        error=ContractValidationError("source", tuple(source_errors)),
                        ledger=ledger,
                    )
                self.artifacts.event(WorkflowState.ASSESSMENT_SOURCE_CHECKS_VALIDATED.value)
            state = WorkflowState.ASSESSMENT_VALIDATED
            verdict = assessment["verdict"]
            self.artifacts.event(state.value, verdict=verdict)

            assessment_sha = self.artifacts.digest_file("reviewer/patch_assessment.json")
            if self.config.contract_version in {"2.0", "3.0"}:
                effective_review_set = self.validator.build_effective_review_set(
                    task_id=task.task_id,
                    ledger_sha256=ledger_sha,
                    assessment_sha256=assessment_sha,
                    ledger=ledger,
                    assessment=assessment,
                )
                self.artifacts.write_json(
                    "reviewer/effective_review_set.json", effective_review_set
                )
                self._effective_review_set_sha256 = self.artifacts.digest_file(
                    "reviewer/effective_review_set.json"
                )
                unresolved_key = (
                    "unresolved_criteria"
                    if self.config.contract_version == "2.0"
                    else "unresolved_obligations"
                )
                self._unresolved_review_items = effective_review_set[unresolved_key]
                unresolved_payload = {unresolved_key: self._unresolved_review_items}
                self.artifacts.write_json(
                    "reviewer/routing_validation.json",
                    {
                        "schema_version": "1.0",
                        "valid": True,
                        "verdict": verdict,
                        "repair_request_ids": effective_review_set["repair_request_ids"],
                        **unresolved_payload,
                        "effective_review_set_sha256": self._effective_review_set_sha256,
                    },
                )
                self.artifacts.event(
                    WorkflowState.EFFECTIVE_REVIEW_SET_FROZEN.value,
                    sha256=self._effective_review_set_sha256,
                )

            p1 = p0
            if verdict in {"approve", "abstain"}:
                state = WorkflowState.REPAIR_SKIPPED
                reason = "approve" if verdict == "approve" else "abstain"
                if verdict == "abstain":
                    self._abstain_reason = (
                        assessment.get("verdict_rationale")
                        if self.config.contract_version == "3.0"
                        else assessment.get("abstain_reason")
                    )
                    self.artifacts.event(
                        WorkflowState.REVIEW_ABSTAINED.value,
                        reason=self._abstain_reason,
                    )
                self.artifacts.event(state.value, reason=reason)
            else:
                repair_executed = True
                state = WorkflowState.REPAIR_RUNNING
                self.artifacts.event(state.value, verdict=verdict)
                await self.workspace.apply_patch(worktree=worktrees.repair, patch=p0)
                audit_path = "not-required"
                audit_schema_path = "not-required"
                effective_set_path = "not-applicable"
                if self.config.contract_version in {"2.0", "3.0"}:
                    effective_set_path = await self.workspace.materialize_file(
                        name="effective_review_set.json",
                        payload=(
                            self.artifacts.root / "reviewer/effective_review_set.json"
                        ).read_bytes(),
                        readonly=True,
                    )
                    if self.config.repair_audit_required:
                        audit_path = await self.workspace.materialize_file(
                            name="repair_audit.json", payload=b"", readonly=False
                        )
                        audit_schema_name = (
                            "repair_audit_v2.schema.json"
                            if self.config.contract_version == "2.0"
                            else "repair_audit_v3.schema.json"
                        )
                        audit_schema_path = await self.workspace.materialize_file(
                            name=audit_schema_name,
                            payload=(self.config.schema_dir / audit_schema_name).read_bytes(),
                            readonly=True,
                        )
                        self._repair_audit_status = "required"
                    else:
                        self._repair_audit_status = "not_required"
                repair_input = self._repair_input(
                    task,
                    worktrees.repair,
                    p0_sha,
                    ledger,
                    assessment,
                    effective_review_set=effective_review_set,
                    assessment_sha=assessment_sha,
                    audit_path=audit_path,
                    audit_schema_path=audit_schema_path,
                )
                self.validator.validate_schema("repair_input", repair_input)
                self.artifacts.write_json("inputs/repair_input.json", repair_input)
                repair_prompt, repair_hash = self.renderer.render(
                    self.config.prompt_templates["repair_once"],
                    {
                        "REPAIR_INPUT_JSON": self.renderer.json_text(repair_input),
                        "REPAIR_FEW_SHOT_EXAMPLES": self.renderer.load_few_shots(
                            self.config.repair_few_shots
                        ),
                    },
                )
                self._prompt_hashes["repair_once"] = repair_hash
                self.artifacts.write_text("prompts/repair_once.rendered.md", repair_prompt)
                repair_result = await self._run_role(
                    role=RoleName.REPAIR,
                    prompt=repair_prompt,
                    worktree=worktrees.repair,
                    prefix="repair",
                )
                if self.config.repair_audit_required:
                    observation = observe_repair_sequence(
                        repair_result.trajectory_path,
                        audit_schema_path=audit_schema_path,
                        audit_output_path=audit_path,
                        frozen_inputs={
                            "p0_patch": p0_container_path,
                            "review_obligations": ledger_container_path,
                            "effective_review_set": effective_set_path,
                            "source_policy": source_policy_path or "",
                            "source_bindings": source_bindings_container_path or "",
                        },
                    )
                    self.artifacts.write_json(
                        "repair/repair_sequence_observation.json", observation
                    )
                    self._repair_sequence_observation_status = observation["status"]
                    self._repair_sequence_observation_sha256 = self.artifacts.digest_file(
                        "repair/repair_sequence_observation.json"
                    )
                try:
                    await self._require_reviewer_diff(
                        worktree=worktrees.patch_review,
                        base_commit=task.base_commit,
                        expected=expected_p0_digest,
                        role="Repair",
                    )
                    await self._require_pristine_worktrees(
                        worktrees=(worktrees.spec_review, worktrees.final),
                        base_commit=task.base_commit,
                        role="Repair",
                    )
                    await self._require_materialized_file_unchanged(
                        path=p0_container_path,
                        expected=p0_sha,
                        role="Repair",
                    )
                    await self._require_materialized_file_unchanged(
                        path=ledger_container_path,
                        expected=ledger_sha,
                        role="Repair",
                    )
                    if self.config.contract_version in {"2.0", "3.0"}:
                        await self._require_materialized_file_unchanged(
                            path=effective_set_path,
                            expected=self._effective_review_set_sha256,
                            role="Repair",
                        )
                        if source_policy_path is not None:
                            await self._require_materialized_file_unchanged(
                                path=source_policy_path,
                                expected=self._source_policy_sha256,
                                role="Repair",
                            )
                        if source_bindings_container_path is not None:
                            await self._require_materialized_file_unchanged(
                                path=source_bindings_container_path,
                                expected=self._source_bindings_sha256,
                                role="Repair",
                            )
                        if self.config.repair_audit_required:
                            audit_schema_name = (
                                "repair_audit_v2.schema.json"
                                if self.config.contract_version == "2.0"
                                else "repair_audit_v3.schema.json"
                            )
                            await self._require_materialized_file_unchanged(
                                path=audit_schema_path,
                                expected=sha256_bytes(
                                    (self.config.schema_dir / audit_schema_name).read_bytes()
                                ),
                                role="Repair",
                            )
                except ProtocolViolationError as error:
                    return await self._fallback(
                        state=WorkflowState.PROTOCOL_VIOLATION,
                        task=task,
                        p0=p0,
                        p0_sha=p0_sha,
                        error=error,
                        ledger=ledger,
                        assessment=assessment,
                        verdict=verdict,
                        repair_executed=True,
                    )
                if repair_result.exit_code == 0 and not repair_result.timed_out:
                    p1 = await self.workspace.capture_patch(
                        worktree=worktrees.repair,
                        base_commit=task.base_commit,
                    )
                    if self.config.repair_audit_required:
                        try:
                            repair_audit = await self._capture_and_validate_repair_audit(
                                task=task,
                                worktree=worktrees.spec_review,
                                audit_path=audit_path,
                                p0=p0,
                                p0_sha=p0_sha,
                                p1=p1,
                                assessment_sha=assessment_sha,
                                ledger=ledger,
                                assessment=assessment,
                                effective_review_set=effective_review_set,
                            )
                        except (ContractValidationError, OSError, ValueError) as error:
                            self._repair_audit_status = "failed"
                            self.artifacts.write_json(
                                "repair/repair_audit_validation.json",
                                {
                                    "schema_version": "1.0",
                                    "valid": False,
                                    "errors": (
                                        error.as_json()
                                        if isinstance(error, ContractValidationError)
                                        else [{"kind": "audit", "message": str(error)}]
                                    ),
                                },
                            )
                            if p1 != p0:
                                self.artifacts.write_bytes("repair/discarded_candidate.patch", p1)
                            return await self._fallback(
                                state=WorkflowState.REPAIR_AUDIT_FAILED,
                                task=task,
                                p0=p0,
                                p0_sha=p0_sha,
                                error=error,
                                ledger=ledger,
                                assessment=assessment,
                                effective_review_set=effective_review_set,
                                verdict=verdict,
                                repair_executed=True,
                            )
                else:
                    repair_error = repair_result.exception or (
                        f"exit code {repair_result.exit_code}"
                    )
                    try:
                        p1 = await self.workspace.capture_patch(
                            worktree=worktrees.repair,
                            base_commit=task.base_commit,
                        )
                    except Exception:  # noqa: BLE001 - invalid partial diff falls back to P0
                        p1 = p0
                    self.artifacts.event(
                        WorkflowState.REPAIR_FAILED.value,
                        error=repair_error,
                        partial_patch_captured=p1 != p0,
                    )
                    if self.config.repair_audit_required:
                        await self._capture_incomplete_repair_audit(audit_path)
            self._write_run_metadata_artifacts()
            await self.workspace.publish_final_patch(base_commit=task.base_commit, patch=p1)
            p1_sha = sha256_bytes(p1)
            self.artifacts.write_bytes("outputs/p1.patch", p1)
            self.artifacts.write_json(
                "outputs/patch_hashes.json",
                {"schema_version": "1.0", "p0_sha256": p0_sha, "p1_sha256": p1_sha},
            )
            state = WorkflowState.P1_CAPTURED
            self.artifacts.event(state.value, p1_sha256=p1_sha)
            state = WorkflowState.REPAIR_FAILED if repair_error else WorkflowState.COMPLETED
            self.artifacts.event(state.value)
            self._write_manifest(
                task=task,
                state=state,
                p0_sha=p0_sha,
                p1_sha=p1_sha,
                verdict=verdict,
                repair_executed=repair_executed,
                existing_p0=existing_p0,
                error=repair_error,
            )
            return WorkflowResult(
                state=state,
                p0_sha256=p0_sha,
                p1_sha256=p1_sha,
                p1_patch=p1,
                verdict=verdict,
                repair_executed=repair_executed,
                ledger=ledger,
                assessment=assessment,
                effective_review_set=effective_review_set,
                repair_audit=repair_audit,
                error=repair_error,
            )
        finally:
            await self.workspace.cleanup()

    def _load_existing_p0(self, task: TaskInput, existing: ExistingPatch) -> bytes:
        if existing.task_id != task.task_id:
            raise ValueError("P0 task_id does not match the active task")
        if not git_commits_equivalent(existing.base_commit, task.base_commit):
            raise ValueError("P0 base commit does not match the active repository")
        patch = existing.path.read_bytes()
        if sha256_bytes(patch) != existing.sha256:
            raise ValueError("P0 file digest does not match its manifest")
        return patch

    async def _run_role(
        self, *, role: RoleName, prompt: str, worktree: str, prefix: str
    ) -> RoleRunResult:
        result = await self.role_runner.run_role(
            role=role,
            prompt=prompt,
            worktree=worktree,
            tools=self.config.tools_for(role),
            timeout_sec=self.config.timeout_for(role),
            artifact_prefix=prefix,
        )
        self._role_results.append((role, result))
        return result

    @staticmethod
    def _require_role_success(result: RoleRunResult, label: str) -> None:
        if result.timed_out:
            raise RuntimeError(f"{label} timed out")
        if result.exit_code != 0:
            raise RuntimeError(
                f"{label} exited with status {result.exit_code}: {result.exception or 'no detail'}"
            )

    async def _require_reviewer_unchanged(self, worktree: str, base_commit: str, role: str) -> None:
        digest = await self.workspace.tracked_diff_digest(
            worktree=worktree, base_commit=base_commit
        )
        if digest != EMPTY_DIFF_SHA256:
            raise ProtocolViolationError(f"{role} modified tracked repository files")

    async def _require_reviewer_tracked_files_unchanged(
        self, worktree: str, base_commit: str, role: str
    ) -> None:
        digest = await self.workspace.tracked_files_diff_digest(
            worktree=worktree,
            base_commit=base_commit,
        )
        if digest != EMPTY_DIFF_SHA256:
            raise ProtocolViolationError(f"{role} modified base-tracked repository files")

    async def _restore_spec_reviewer_worktree(
        self, *, worktree: str, base_commit: str
    ) -> None:
        discarded = await self.workspace.restore_reviewer_worktree(
            worktree=worktree,
            base_commit=base_commit,
        )
        if not discarded:
            return
        artifact_path = "reviewer/stage_a_discarded_untracked.patch"
        self.artifacts.write_bytes(artifact_path, discarded)
        self._spec_reviewer_discarded_patch_sha256 = sha256_bytes(discarded)
        self.artifacts.event(
            "SPECIFICATION_REVIEWER_UNTRACKED_DISCARDED",
            artifact=artifact_path,
            discarded_patch_sha256=self._spec_reviewer_discarded_patch_sha256,
        )

    async def _require_reviewer_diff(
        self, *, worktree: str, base_commit: str, expected: str, role: str
    ) -> None:
        digest = await self.workspace.tracked_diff_digest(
            worktree=worktree, base_commit=base_commit
        )
        if digest != expected:
            raise ProtocolViolationError(f"{role} modified the frozen P0 worktree")

    async def _require_patch_reviewer_frozen_p0(
        self,
        *,
        worktree: str,
        base_commit: str,
        expected_tracked_digest: str,
        p0: bytes,
        role: str,
    ) -> None:
        """Protect every base-tracked and P0-owned path while allowing new scratch files."""
        tracked_digest = await self.workspace.tracked_files_diff_digest(
            worktree=worktree,
            base_commit=base_commit,
        )
        if tracked_digest != expected_tracked_digest:
            raise ProtocolViolationError(f"{role} modified base-tracked repository files")
        current = await self.workspace.capture_patch(
            worktree=worktree,
            base_commit=base_commit,
        )
        modified_p0_paths = sorted(
            changed_paths_between_patches(p0, current) & patch_changed_paths(p0)
        )
        if modified_p0_paths:
            raise ProtocolViolationError(
                f"{role} modified P0-owned repository paths: " + ", ".join(modified_p0_paths)
            )

    async def _restore_patch_reviewer_worktree(
        self,
        *,
        worktree: str,
        base_commit: str,
        p0: bytes,
        expected_p0_digest: str,
    ) -> None:
        """Remove reviewer scratch output, then reconstruct the exact frozen P0 view."""
        before = await self.workspace.capture_patch(worktree=worktree, base_commit=base_commit)
        scratch_paths = sorted(changed_paths_between_patches(p0, before) - patch_changed_paths(p0))
        await self.workspace.restore_reviewer_worktree(
            worktree=worktree,
            base_commit=base_commit,
        )
        await self.workspace.apply_patch(worktree=worktree, patch=p0)
        restored_digest = await self.workspace.tracked_diff_digest(
            worktree=worktree,
            base_commit=base_commit,
        )
        if restored_digest != expected_p0_digest:
            raise ProtocolViolationError(
                "Patch Reviewer scratch cleanup did not restore the frozen P0 worktree"
            )
        if scratch_paths:
            self.artifacts.event(
                "PATCH_REVIEWER_UNTRACKED_DISCARDED",
                discarded_paths=scratch_paths,
            )

    async def _require_pristine_worktrees(
        self,
        *,
        worktrees: tuple[str, ...],
        base_commit: str,
        role: str,
    ) -> None:
        for worktree in worktrees:
            digest = await self.workspace.tracked_diff_digest(
                worktree=worktree,
                base_commit=base_commit,
            )
            if digest != EMPTY_DIFF_SHA256:
                raise ProtocolViolationError(
                    f"{role} modified an out-of-scope worktree: {worktree}"
                )

    async def _require_materialized_file_unchanged(
        self, *, path: str, expected: str, role: str
    ) -> None:
        digest = await self.workspace.materialized_file_digest(path=path)
        if digest != expected:
            raise ProtocolViolationError(f"{role} modified frozen input {path}")

    async def _validate_with_corrections(
        self,
        *,
        raw: str,
        schema_name: str,
        semantic_validate,
        worktree: str,
        prefix: str,
        async_validate=None,
    ) -> dict[str, Any]:
        current, normalization_method = extract_single_json_object(raw)
        normalization_methods = [normalization_method] if normalization_method is not None else []
        last_error: ContractValidationError | None = None
        protected_semantics_sha256: str | None = None
        for attempt in range(self.config.max_json_corrections + 1):
            try:
                payload = self.validator.parse(current)
                if self.config.contract_version in {"2.0", "3.0"}:
                    current_semantics_sha256 = self._correction_semantics_sha256(
                        schema_name, payload
                    )
                    if protected_semantics_sha256 is None:
                        protected_semantics_sha256 = current_semantics_sha256
                    elif current_semantics_sha256 != protected_semantics_sha256:
                        raise ContractValidationError(
                            "protocol",
                            (
                                (
                                    "JSON Corrector changed protected "
                                    f"v{self.config.contract_version.split('.')[0]} semantic "
                                    "fields; semantic repair requires a new Reviewer decision"
                                ),
                            ),
                        )
                semantic_validate(payload)
                if async_validate is not None:
                    async_errors = await async_validate(payload)
                    if async_errors:
                        raise ContractValidationError("semantic", tuple(async_errors))
                self.artifacts.write_json(
                    f"{prefix.rsplit('_correction', 1)[0]}_validation.json",
                    {
                        "valid": True,
                        "correction_used": attempt > 0,
                        "correction_count": attempt,
                        "normalization_used": bool(normalization_methods),
                        "normalization_method": (
                            normalization_methods[-1] if normalization_methods else None
                        ),
                        "normalization_methods": normalization_methods,
                        "payload_sha256": sha256_json(payload),
                    },
                )
                return payload
            except ContractValidationError as error:
                last_error = error
                if error.kind == "protocol":
                    self.artifacts.write_json(
                        f"{prefix.rsplit('_correction', 1)[0]}_validation.json",
                        {
                            "valid": False,
                            "correction_used": attempt > 0,
                            "correction_count": attempt,
                            "normalization_used": bool(normalization_methods),
                            "normalization_method": (
                                normalization_methods[-1] if normalization_methods else None
                            ),
                            "normalization_methods": normalization_methods,
                            "errors": error.as_json(),
                        },
                    )
                    raise
                if attempt >= self.config.max_json_corrections:
                    break
                correction_prompt, _ = self.renderer.render(
                    self.config.prompt_templates["correct_json"],
                    {
                        "JSON_SCHEMA": self.renderer.json_text(self.validator.schema(schema_name)),
                        "VALIDATION_ERRORS_JSON": self.renderer.json_text(error.as_json()),
                        "ORIGINAL_OUTPUT": current,
                    },
                )
                self._prompt_hashes[f"{prefix}_{attempt + 1}"] = sha256_bytes(
                    correction_prompt.encode("utf-8")
                )
                correction = await self._run_role(
                    role=RoleName.JSON_CORRECTOR,
                    prompt=correction_prompt,
                    worktree=worktree,
                    prefix=f"{prefix}_{attempt + 1}",
                )
                self._require_role_success(correction, "JSON Corrector")
                current, normalization_method = extract_single_json_object(
                    correction.final_assistant_text
                )
                if normalization_method is not None:
                    normalization_methods.append(normalization_method)
        assert last_error is not None
        self.artifacts.write_json(
            f"{prefix.rsplit('_correction', 1)[0]}_validation.json",
            {
                "valid": False,
                "correction_used": True,
                "correction_count": self.config.max_json_corrections,
                "normalization_used": bool(normalization_methods),
                "normalization_method": (
                    normalization_methods[-1] if normalization_methods else None
                ),
                "normalization_methods": normalization_methods,
                "errors": last_error.as_json(),
            },
        )
        raise last_error

    def _correction_semantics_sha256(self, schema_name: str, payload: dict[str, Any]) -> str:
        """Freeze semantic contract fields across parseable correction attempts.

        A corrector may remove unknown wrapper fields or repair JSON syntax, but it
        cannot rewrite a parseable Reviewer's sources, findings, verdict, or requests.
        V3 recursively ignores schema-unknown fields and normalizes only neutral missing
        values that the contract uniquely determines. Other missing recognized fields
        remain explicit so a corrector cannot manufacture semantic content.
        """
        if self.config.contract_version == "3.0":
            projection = _project_v3_correction_semantics(schema_name, payload)
            if projection is not None:
                return sha256_json(projection)
        known_fields = sorted(self.validator.schema(schema_name).get("properties", {}))
        projection = [
            {
                "field": field,
                "present": field in payload,
                "value": payload.get(field),
            }
            for field in known_fields
        ]
        return sha256_json(projection)

    async def _fallback(
        self,
        *,
        state: WorkflowState,
        task: TaskInput,
        p0: bytes,
        p0_sha: str,
        error: Exception,
        ledger: dict[str, Any] | None = None,
        assessment: dict[str, Any] | None = None,
        effective_review_set: dict[str, Any] | None = None,
        repair_audit: dict[str, Any] | None = None,
        verdict: str | None = None,
        repair_executed: bool = False,
    ) -> WorkflowResult:
        await self._restore_final_worktree(
            base_commit=task.base_commit,
            artifact_path="outputs/discarded_final_worktree.patch",
            event="FINAL_WORKTREE_RESTORED_FOR_FALLBACK",
        )
        await self.workspace.publish_final_patch(base_commit=task.base_commit, patch=p0)
        self.artifacts.write_bytes("outputs/p1.patch", p0)
        self.artifacts.write_json(
            "outputs/patch_hashes.json",
            {"schema_version": "1.0", "p0_sha256": p0_sha, "p1_sha256": p0_sha},
        )
        self.artifacts.event(state.value, error=str(error), fallback="P0")
        self._write_run_metadata_artifacts()
        self._write_manifest(
            task=task,
            state=state,
            p0_sha=p0_sha,
            p1_sha=p0_sha,
            verdict=verdict,
            repair_executed=repair_executed,
            existing_p0=self._existing_p0,
            error=str(error),
        )
        return WorkflowResult(
            state=state,
            p0_sha256=p0_sha,
            p1_sha256=p0_sha,
            p1_patch=p0,
            verdict=verdict,
            repair_executed=repair_executed,
            ledger=ledger,
            assessment=assessment,
            effective_review_set=effective_review_set,
            repair_audit=repair_audit,
            error=str(error),
        )

    async def _restore_final_worktree(
        self,
        *,
        base_commit: str,
        artifact_path: str,
        event: str,
    ) -> None:
        discarded = await self.workspace.restore_final_worktree(base_commit=base_commit)
        details: dict[str, Any] = {
            "discarded_changes": bool(discarded),
            "discarded_patch_sha256": sha256_bytes(discarded),
        }
        if discarded:
            self.artifacts.write_bytes(artifact_path, discarded)
            details["artifact"] = artifact_path
        self.artifacts.event(event, **details)

    async def _source_anchor_errors(
        self,
        *,
        task: TaskInput,
        worktree: str,
        sources: list[dict[str, Any]],
    ) -> list[str]:
        if self.config.contract_version != "2.0":
            return []
        errors: list[str] = []
        try:
            self.validator.validate_issue_source_anchors(
                sources,
                title=task.issue_title,
                body=task.issue_body,
            )
        except ContractValidationError as error:
            errors.extend(error.errors)
        errors.extend(
            await self.workspace.validate_repository_sources(
                worktree=worktree,
                base_commit=task.base_commit,
                sources=sources,
            )
        )
        return errors

    async def _record_ledger_source_validation(
        self,
        *,
        task: TaskInput,
        worktree: str,
        sources: list[dict[str, Any]],
    ) -> list[str]:
        if self.config.contract_version == "3.0":
            errors = await self._record_v3_source_bindings(
                task=task,
                worktree=worktree,
                sources=sources,
                artifact_path="reviewer/source_bindings.json",
            )
            if not errors:
                self._source_bindings_sha256 = self.artifacts.digest_file(
                    "reviewer/source_bindings.json"
                )
        else:
            errors = await self._source_anchor_errors(
                task=task,
                worktree=worktree,
                sources=sources,
            )
        self._write_source_anchor_validation(
            ledger_source_ids=[item.get("id", "") for item in sources],
            additional_source_ids=[],
            errors=errors,
        )
        return errors

    async def _record_v3_source_bindings(
        self,
        *,
        task: TaskInput,
        worktree: str,
        sources: list[dict[str, Any]],
        artifact_path: str,
    ) -> list[str]:
        errors: list[str] = []
        try:
            self.validator.validate_task_source_anchors(
                sources,
                title=task.issue_title,
                description=task.issue_body,
            )
        except ContractValidationError as error:
            errors.extend(error.errors)
        bound, repository_errors = await self.workspace.bind_repository_sources(
            worktree=worktree,
            base_commit=task.base_commit,
            sources=sources,
        )
        errors.extend(repository_errors)
        if errors:
            return errors
        payload = {
            "schema_version": "3.0",
            "task_id": task.task_id,
            "base_commit": task.base_commit,
            "sources": bound,
        }
        try:
            self.validator.validate_source_bindings(payload)
        except ContractValidationError as error:
            errors.extend(error.errors)
        if not errors:
            self.artifacts.write_json(artifact_path, payload)
        return errors

    async def _record_assessment_source_validation(
        self,
        *,
        task: TaskInput,
        worktree: str,
        ledger: dict[str, Any],
        assessment: dict[str, Any],
    ) -> list[str]:
        sources = assessment.get("additional_sources", [])
        if self.config.contract_version == "3.0":
            errors = await self._record_v3_source_bindings(
                task=task,
                worktree=worktree,
                sources=sources,
                artifact_path="reviewer/additional_source_bindings.json",
            )
        else:
            errors = await self._source_anchor_errors(
                task=task,
                worktree=worktree,
                sources=sources,
            )
        self._write_source_anchor_validation(
            ledger_source_ids=[item["id"] for item in ledger["sources"]],
            additional_source_ids=[item.get("id", "") for item in sources],
            errors=errors,
        )
        return errors

    def _write_source_anchor_validation(
        self,
        *,
        ledger_source_ids: list[str],
        additional_source_ids: list[str],
        errors: list[str],
    ) -> None:
        self.artifacts.write_json(
            "reviewer/source_anchor_validation.json",
            {
                "schema_version": "1.0",
                "valid": not errors,
                "ledger_source_ids": ledger_source_ids,
                "additional_source_ids": additional_source_ids,
                "source_policy_sha256": self._source_policy_sha256,
                "errors": [{"kind": "source", "message": message} for message in errors],
            },
        )

    async def _assessment_validation_errors(
        self,
        *,
        task: TaskInput,
        evidence_worktree: str,
        source_worktree: str,
        ledger: dict[str, Any],
        assessment: dict[str, Any],
    ) -> list[str]:
        evidence_errors = await self.workspace.validate_evidence(
            worktree=evidence_worktree,
            assessment=assessment,
        )
        if self.config.contract_version == "1.0":
            return evidence_errors
        source_errors = await self._record_assessment_source_validation(
            task=task,
            worktree=source_worktree,
            ledger=ledger,
            assessment=assessment,
        )
        return [*evidence_errors, *source_errors]

    async def _capture_and_validate_repair_audit(
        self,
        *,
        task: TaskInput,
        worktree: str,
        audit_path: str,
        p0: bytes,
        p0_sha: str,
        p1: bytes,
        assessment_sha: str,
        ledger: dict[str, Any],
        assessment: dict[str, Any],
        effective_review_set: dict[str, Any] | None,
    ) -> dict[str, Any]:
        assert effective_review_set is not None
        assert self._effective_review_set_sha256 is not None
        raw = await self.workspace.read_materialized_file(path=audit_path)
        self.artifacts.write_bytes("repair/repair_audit.raw.json", raw)
        if not raw.strip():
            raise ContractValidationError("schema", ("Repair audit file is empty",))
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContractValidationError("schema", ("Repair audit is not valid UTF-8",)) from error
        payload = self.validator.parse(text)
        self.validator.validate_repair_audit(
            payload,
            task_id=task.task_id,
            p0_sha256=p0_sha,
            assessment_sha256=assessment_sha,
            effective_review_set_sha256=self._effective_review_set_sha256,
            ledger=ledger,
            assessment=assessment,
            effective_review_set=effective_review_set,
            actual_changed_paths=changed_paths_between_patches(p0, p1),
        )
        if self.config.contract_version == "2.0":
            source_errors = await self._source_anchor_errors(
                task=task,
                worktree=worktree,
                sources=payload["additional_sources"],
            )
            if source_errors:
                raise ContractValidationError("source", tuple(source_errors))
        self.artifacts.write_json("repair/repair_audit.json", payload)
        self.artifacts.write_json(
            "repair/repair_audit_validation.json",
            {
                "schema_version": "1.0",
                "valid": True,
                "payload_sha256": sha256_json(payload),
                "changed_paths": payload["changed_paths"],
            },
        )
        self._repair_audit_status = "validated"
        self.artifacts.event(WorkflowState.REPAIR_AUDIT_VALIDATED.value)
        return payload

    async def _capture_incomplete_repair_audit(self, audit_path: str) -> None:
        try:
            raw = await self.workspace.read_materialized_file(path=audit_path)
        except Exception:  # noqa: BLE001 - timeout diagnostics are best effort
            raw = b""
        if raw:
            self.artifacts.write_bytes("repair/repair_audit.raw.json", raw)
        self.artifacts.write_json(
            "repair/repair_audit_validation.json",
            {
                "schema_version": "1.0",
                "valid": False,
                "incomplete": True,
                "errors": [
                    {
                        "kind": "timeout",
                        "message": "Repair did not finish; audit completeness was not asserted",
                    }
                ],
            },
        )
        self._repair_audit_status = "incomplete"
        self.artifacts.event(WorkflowState.REPAIR_AUDIT_INCOMPLETE.value)

    def _source_policy(self, task: TaskInput) -> dict[str, Any]:
        return {
            "schema_version": self.config.source_policy_version,
            "contract_version": "2.0",
            "task_id": task.task_id,
            "issue_sha256": sha256_json({"title": task.issue_title, "body": task.issue_body}),
            "base_commit": task.base_commit,
            "limits": {
                "max_obligations": self.config.max_obligations,
                "max_criteria_per_obligation": self.config.max_criteria_per_obligation,
                "max_sources": 128,
                "max_challenges": 32,
                "max_repair_requests": 3,
            },
            "allowed_basis_kinds": [
                "explicit_issue",
                "repository_contract",
                "compatibility_inferred",
                "unspecified",
            ],
            "forbid_candidate_as_source": True,
            "forbid_hidden_evidence": True,
        }

    def _build_obligations_input(
        self,
        task: TaskInput,
        worktree: str,
        *,
        source_policy_path: str | None,
    ) -> dict[str, Any]:
        common = {
            "schema_version": "1.0",
            "task_id": task.task_id,
            "issue": {"title": task.issue_title, "body": task.issue_body},
            "repository": {
                "name": task.repository_name,
                "base_commit": task.base_commit,
                "worktree": worktree,
                "state": "base_only",
            },
            "policy": {
                "max_obligations": 32,
                "allowed_origin_types": [
                    "explicit_issue",
                    "compatibility_inferred",
                    "repository_contract",
                ],
                "forbid_patch_access": True,
                "forbid_hidden_evidence": True,
            },
        }
        if self.config.contract_version == "1.0":
            return common
        if self.config.contract_version == "3.0":
            return {
                "schema_version": "3.0",
                "task_id": task.task_id,
                "task": {
                    "title": task.issue_title,
                    "description": task.issue_body,
                },
                "repository": common["repository"],
            }
        assert source_policy_path is not None and self._source_policy_sha256 is not None
        return {
            "schema_version": "2.0",
            "task_id": task.task_id,
            "issue": common["issue"],
            "repository": common["repository"],
            "source_policy": {
                "path": source_policy_path,
                "sha256": self._source_policy_sha256,
            },
        }

    def _assess_patch_input(
        self,
        task: TaskInput,
        worktree: str,
        p0_sha: str,
        ledger_sha: str,
        p0_path: str,
        ledger_path: str,
        *,
        source_policy_path: str | None,
        source_bindings_path: str | None,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "task_id": task.task_id,
            "issue": {"title": task.issue_title, "body": task.issue_body},
            "repository": {
                "name": task.repository_name,
                "base_commit": task.base_commit,
                "worktree": worktree,
                "state": "base_plus_p0",
            },
            "patch": {"path": p0_path, "sha256": p0_sha},
            "obligation_ledger": {
                "path": ledger_path,
                "sha256": ledger_sha,
            },
            "public_test_results": [],
        }
        if self.config.contract_version == "2.0":
            assert source_policy_path is not None and self._source_policy_sha256 is not None
            payload["schema_version"] = "2.0"
            payload["source_policy"] = {
                "path": source_policy_path,
                "sha256": self._source_policy_sha256,
            }
        elif self.config.contract_version == "3.0":
            assert source_bindings_path is not None
            assert self._source_bindings_sha256 is not None
            payload.update(
                {
                    "schema_version": "3.0",
                    "task": {
                        "title": task.issue_title,
                        "description": task.issue_body,
                    },
                    "source_bindings": {
                        "path": source_bindings_path,
                        "sha256": self._source_bindings_sha256,
                    },
                }
            )
            payload.pop("issue")
            payload["repository"]["state"] = "base_plus_candidate"
        return payload

    def _repair_input(
        self,
        task: TaskInput,
        worktree: str,
        p0_sha: str,
        ledger: dict[str, Any],
        assessment: dict[str, Any],
        *,
        effective_review_set: dict[str, Any] | None,
        assessment_sha: str,
        audit_path: str,
        audit_schema_path: str,
    ) -> dict[str, Any]:
        common = {
            "schema_version": "1.0",
            "task_id": task.task_id,
            "issue": {"title": task.issue_title, "body": task.issue_body},
            "repository": {
                "base_commit": task.base_commit,
                "worktree": worktree,
                "state": "base_plus_p0",
            },
            "patch": {"sha256": p0_sha},
            "review_obligations": ledger,
            "patch_assessment": assessment,
            "policy": {"max_repair_rounds": 1, "hidden_evidence_forbidden": True},
        }
        if self.config.contract_version == "1.0":
            return common
        assert effective_review_set is not None
        assert self._effective_review_set_sha256 is not None
        if self.config.contract_version == "3.0":
            ledger_bindings = self.validator.parse(
                (self.artifacts.root / "reviewer/source_bindings.json").read_text(
                    encoding="utf-8"
                )
            )
            additional_bindings = self.validator.parse(
                (
                    self.artifacts.root / "reviewer/additional_source_bindings.json"
                ).read_text(encoding="utf-8")
            )
            source_bindings = {
                "schema_version": "3.0",
                "task_id": task.task_id,
                "base_commit": task.base_commit,
                "sources": [
                    *ledger_bindings["sources"],
                    *additional_bindings["sources"],
                ],
            }
            self.validator.validate_source_bindings(source_bindings)
            return {
                "schema_version": "3.0",
                "task_id": task.task_id,
                "task": {
                    "title": task.issue_title,
                    "description": task.issue_body,
                },
                "repository": {
                    "base_commit": task.base_commit,
                    "worktree": worktree,
                    "state": "base_plus_candidate",
                },
                "candidate_patch": {"sha256": p0_sha},
                "review_obligations": ledger,
                "patch_assessment": assessment,
                "effective_review_set": effective_review_set,
                "request_contexts": self.validator.build_request_contexts(
                    ledger=ledger,
                    assessment=assessment,
                ),
                "source_bindings": source_bindings,
                "audit": {
                    "required": self.config.repair_audit_required,
                    "path": audit_path,
                    "schema_path": audit_schema_path,
                    "assessment_sha256": assessment_sha,
                    "effective_review_set_sha256": self._effective_review_set_sha256,
                },
                "policy": {
                    "max_repair_rounds": 1,
                    "hidden_evidence_forbidden": True,
                    "changes_outside_requests_require_audit": (
                        self.config.repair_audit_required
                    ),
                },
            }
        common.update(
            {
                "schema_version": "2.0",
                "effective_review_set": effective_review_set,
                "audit": {
                    "required": self.config.repair_audit_required,
                    "path": audit_path,
                    "schema_path": audit_schema_path,
                    "assessment_sha256": assessment_sha,
                    "effective_review_set_sha256": self._effective_review_set_sha256,
                },
                "policy": {
                    "max_repair_rounds": 1,
                    "hidden_evidence_forbidden": True,
                    "high_risk_source_check_required": self.config.repair_audit_required,
                    "changes_outside_requests_require_audit": self.config.repair_audit_required,
                },
            }
        )
        return common

    def _write_manifest(
        self,
        *,
        task: TaskInput,
        state: WorkflowState,
        p0_sha: str,
        p1_sha: str,
        verdict: str | None,
        repair_executed: bool,
        existing_p0: ExistingPatch | None,
        error: str | None = None,
    ) -> None:
        self.artifacts.write_json(
            "manifest.json",
            {
                "schema_version": "1.0",
                "task_id": task.task_id,
                "benchmark_revision": "1.1",
                "base_commit": task.base_commit,
                "source_job_id": existing_p0.source_job_id if existing_p0 else None,
                "source_trial_id": existing_p0.source_trial_id if existing_p0 else None,
                "p0_sha256": p0_sha,
                "p1_sha256": p1_sha,
                "contract_version": self.config.contract_version,
                "source_policy_version": (
                    self.config.source_policy_version
                    if self.config.contract_version == "2.0"
                    else None
                ),
                "experiment_arm": self.config.experiment_arm,
                "prompt_version": self.config.prompt_version,
                "controller_version": self.config.controller_version,
                "model": self.config.model,
                "agent_version": self.config.agent_version,
                "tool_profiles": {
                    role.value: list(self.config.tools_for(role)) for role in RoleName
                },
                "prompt_sha256": dict(self._prompt_hashes),
                "prompt_template_sha256": {
                    name: sha256_bytes((self.config.prompt_dir / name).read_bytes())
                    for name in sorted(set(self.config.prompt_templates.values()))
                },
                "schema_sha256": {
                    name: sha256_bytes((self.config.schema_dir / name).read_bytes())
                    for name in sorted(set(self.validator.SCHEMAS.values()))
                },
                "implementation_sha256": {
                    (
                        path.name
                        if self.config.contract_version == "1.0"
                        else self.config.implementation_label(path)
                    ): sha256_bytes(path.read_bytes())
                    for path in self.config.implementation_paths
                },
                "few_shot_sha256": {
                    str(path): sha256_bytes(path.read_bytes())
                    for path in (
                        *self.config.stage_a_few_shots,
                        *self.config.stage_b_few_shots,
                        *self.config.repair_few_shots,
                        *self.config.few_shot_metadata,
                    )
                },
                "few_shot_summary": {
                    "stage_a_artifact_count": len(self.config.stage_a_few_shots),
                    "stage_b_artifact_count": len(self.config.stage_b_few_shots),
                    "repair_artifact_count": len(self.config.repair_few_shots),
                },
                "schema_summary": {
                    "contract_version": self.config.contract_version,
                    "artifacts": dict(sorted(self.validator.SCHEMAS.items())),
                },
                "reviewer_verdict": verdict,
                "abstain_reason": self._abstain_reason,
                "source_policy_sha256": self._source_policy_sha256,
                "source_bindings_sha256": self._source_bindings_sha256,
                "effective_review_set_sha256": self._effective_review_set_sha256,
                "unresolved_criteria": (
                    self._unresolved_review_items
                    if self.config.contract_version != "3.0"
                    else None
                ),
                "unresolved_obligations": (
                    self._unresolved_review_items
                    if self.config.contract_version == "3.0"
                    else None
                ),
                "repair_audit_status": self._repair_audit_status,
                "repair_sequence_observation_status": (self._repair_sequence_observation_status),
                "repair_sequence_observation_sha256": (self._repair_sequence_observation_sha256),
                "spec_reviewer_discarded_untracked_sha256": (
                    self._spec_reviewer_discarded_patch_sha256
                ),
                "repair_executed": repair_executed,
                "final_state": state.value,
                "error": error,
            },
        )

    def _write_run_metadata_artifacts(self) -> None:
        self.artifacts.write_json(
            "prompts/prompt_hashes.json",
            {"schema_version": "1.0", **self._prompt_hashes},
        )
        self.artifacts.write_json(
            "outputs/resource_usage.json",
            {
                "schema_version": "1.0",
                "roles": [
                    {
                        "role": role.value,
                        "started_at": result.started_at,
                        "finished_at": result.finished_at,
                        "exit_code": result.exit_code,
                        "timed_out": result.timed_out,
                        "resource_usage": result.resource_usage,
                    }
                    for role, result in self._role_results
                ],
            },
        )

    def _resume_json(self, validation_path: str, payload_path: str) -> dict[str, Any] | None:
        if not self.config.resume:
            return None
        validation_file = self.artifacts.root / validation_path
        payload_file = self.artifacts.root / payload_path
        if not validation_file.is_file() or not payload_file.is_file():
            return None
        try:
            validation = self.validator.parse(validation_file.read_text(encoding="utf-8"))
            payload = self.validator.parse(payload_file.read_text(encoding="utf-8"))
        except (ContractValidationError, OSError, UnicodeError) as error:
            self.artifacts.event(
                "RESUME_ARTIFACT_REJECTED",
                validation_path=validation_path,
                payload_path=payload_path,
                error=str(error),
            )
            return None
        if validation.get("valid") is not True:
            return None
        if validation.get("payload_sha256") != sha256_json(payload):
            self.artifacts.event(
                "RESUME_ARTIFACT_REJECTED",
                validation_path=validation_path,
                payload_path=payload_path,
                error="payload digest does not match validation record",
            )
            return None
        return payload
