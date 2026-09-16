from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class RoleName(str, Enum):
    SOLVER = "solver"
    SPECIFICATION_REVIEWER = "specification_reviewer"
    PATCH_REVIEWER = "patch_reviewer"
    REPAIR = "repair"
    JSON_CORRECTOR = "json_corrector"
    V5_SPECIFICATION_REVIEWER = "v5_specification_reviewer"
    V5_IMPLEMENTATION_REVIEWER = "v5_implementation_reviewer"
    V5_REGRESSION_REVIEWER = "v5_regression_reviewer"
    V5_SUMMARY_REVIEWER = "v5_summary_reviewer"


class IndeterminateInvocationError(RuntimeError):
    """A prior remote invocation may still be live and must not be duplicated."""


class WorkflowState(str, Enum):
    INITIALIZED = "INITIALIZED"
    P0_LOADED = "P0_LOADED"
    P0_SOLVED = "P0_SOLVED"
    LEDGER_RUNNING = "LEDGER_RUNNING"
    LEDGER_VALIDATED = "LEDGER_VALIDATED"
    LEDGER_SOURCE_ANCHORS_VALIDATED = "LEDGER_SOURCE_ANCHORS_VALIDATED"
    ASSESSMENT_RUNNING = "ASSESSMENT_RUNNING"
    ASSESSMENT_VALIDATED = "ASSESSMENT_VALIDATED"
    ASSESSMENT_SOURCE_CHECKS_VALIDATED = "ASSESSMENT_SOURCE_CHECKS_VALIDATED"
    EFFECTIVE_REVIEW_SET_FROZEN = "EFFECTIVE_REVIEW_SET_FROZEN"
    REVIEW_ABSTAINED = "REVIEW_ABSTAINED"
    REPAIR_SKIPPED = "REPAIR_SKIPPED"
    REPAIR_RUNNING = "REPAIR_RUNNING"
    REPAIR_AUDIT_VALIDATED = "REPAIR_AUDIT_VALIDATED"
    REPAIR_AUDIT_INCOMPLETE = "REPAIR_AUDIT_INCOMPLETE"
    P1_CAPTURED = "P1_CAPTURED"
    COMPLETED = "COMPLETED"
    SOLVER_FAILED = "SOLVER_FAILED"
    P0_LOAD_FAILED = "P0_LOAD_FAILED"
    LEDGER_ROLE_FAILED = "LEDGER_ROLE_FAILED"
    LEDGER_SCHEMA_FAILED = "LEDGER_SCHEMA_FAILED"
    LEDGER_SEMANTIC_FAILED = "LEDGER_SEMANTIC_FAILED"
    ASSESSMENT_ROLE_FAILED = "ASSESSMENT_ROLE_FAILED"
    ASSESSMENT_SCHEMA_FAILED = "ASSESSMENT_SCHEMA_FAILED"
    ASSESSMENT_SEMANTIC_FAILED = "ASSESSMENT_SEMANTIC_FAILED"
    REPAIR_FAILED = "REPAIR_FAILED"
    REPAIR_AUDIT_FAILED = "REPAIR_AUDIT_FAILED"
    PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"


TERMINAL_FAILURE_STATES = frozenset(
    {
        WorkflowState.P0_LOAD_FAILED,
        WorkflowState.SOLVER_FAILED,
        WorkflowState.LEDGER_ROLE_FAILED,
        WorkflowState.LEDGER_SCHEMA_FAILED,
        WorkflowState.LEDGER_SEMANTIC_FAILED,
        WorkflowState.ASSESSMENT_ROLE_FAILED,
        WorkflowState.ASSESSMENT_SCHEMA_FAILED,
        WorkflowState.ASSESSMENT_SEMANTIC_FAILED,
        WorkflowState.REPAIR_FAILED,
        WorkflowState.REPAIR_AUDIT_FAILED,
        WorkflowState.PROTOCOL_VIOLATION,
    }
)


@dataclass(frozen=True)
class TaskInput:
    task_id: str
    issue_body: str
    repository_name: str
    base_commit: str
    issue_title: str = ""
    managed_delivery: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ExistingPatch:
    task_id: str
    base_commit: str
    path: Path
    sha256: str
    tracked_diff_sha256: str
    source_job_id: str
    source_trial_id: str


@dataclass(frozen=True)
class RoleRunResult:
    exit_code: int
    final_assistant_text: str
    transcript_path: Path
    trajectory_path: Path
    started_at: str
    finished_at: str
    resource_usage: dict[str, Any] = field(default_factory=dict)
    timed_out: bool = False
    exception: str | None = None
    tracked_diff_digest: str | None = None
    session_file: Path | None = None
    pre_session_sha256: str | None = None
    post_session_sha256: str | None = None
    invocation_id: str | None = None
    remote_session_file: str | None = None


@dataclass(frozen=True)
class WorkflowWorktrees:
    solver: str
    spec_review: str
    patch_review: str
    repair: str
    correction: str
    final: str
    # A read-only adjudicator should not share a mutable reviewer worktree.
    # Keep this optional for compatibility with legacy controller fixtures.
    summary: str | None = None


@dataclass(frozen=True)
class WorkflowResult:
    state: WorkflowState
    p0_sha256: str
    p1_sha256: str
    p1_patch: bytes
    verdict: str | None
    repair_executed: bool
    ledger: dict[str, Any] | None = None
    assessment: dict[str, Any] | None = None
    effective_review_set: dict[str, Any] | None = None
    repair_audit: dict[str, Any] | None = None
    error: str | None = None


class RoleRunner(Protocol):
    async def run_role(
        self,
        *,
        role: RoleName,
        prompt: str,
        worktree: str,
        tools: tuple[str, ...],
        timeout_sec: float | None,
        artifact_prefix: str,
    ) -> RoleRunResult: ...


class WorkspaceManager(Protocol):
    async def prepare(self, *, base_commit: str) -> WorkflowWorktrees: ...

    async def apply_patch(self, *, worktree: str, patch: bytes) -> None: ...

    async def discard_worktree(self, *, worktree: str) -> None: ...

    async def materialize_file(
        self, *, name: str, payload: bytes, readonly: bool = False
    ) -> str: ...

    async def materialized_file_digest(self, *, path: str) -> str: ...

    async def read_materialized_file(self, *, path: str) -> bytes: ...

    async def capture_patch(
        self,
        *,
        worktree: str,
        base_commit: str,
        baseline_patch: bytes = b"",
    ) -> bytes: ...

    async def tracked_diff_digest(
        self,
        *,
        worktree: str,
        base_commit: str,
        baseline_patch: bytes = b"",
    ) -> str: ...

    async def tracked_files_diff_digest(
        self, *, worktree: str, base_commit: str
    ) -> str: ...

    async def restore_reviewer_worktree(
        self,
        *,
        worktree: str,
        base_commit: str,
        baseline_patch: bytes = b"",
    ) -> bytes: ...

    async def validate_evidence(
        self, *, worktree: str, assessment: dict[str, Any]
    ) -> list[str]: ...

    async def validate_repository_sources(
        self,
        *,
        worktree: str,
        base_commit: str,
        sources: list[dict[str, Any]],
    ) -> list[str]: ...

    async def bind_repository_sources(
        self,
        *,
        worktree: str,
        base_commit: str,
        sources: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]: ...

    async def publish_final_patch(
        self,
        *,
        base_commit: str,
        patch: bytes,
        baseline_patch: bytes = b"",
        baseline_manifest: dict[str, Any] | None = None,
    ) -> None: ...

    async def restore_final_worktree(self, *, base_commit: str) -> bytes: ...

    async def cleanup(self) -> None: ...
