import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.agents.pier.pi_role_runner import ContainerGitWorkspaceManager
from external_agents_eval.workflows.solve_review_repair.artifacts import (
    ArtifactStore,
    sha256_bytes,
)
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
)


class LocalEnvironment:
    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        result = subprocess.run(  # noqa: ASYNC221 - synchronous local integration fixture
            ["bash", "-c", command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return SimpleNamespace(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

    async def upload_file(self, source_path, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)

    async def download_file(self, source_path, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)


def git(repository, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class GitV2RoleRunner:
    def __init__(self, *, audit_mode="valid"):
        self.audit_mode = audit_mode
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
        self.calls.append(role)
        if role == RoleName.SPECIFICATION_REVIEWER:
            build_input = json.loads(
                prompt.split("TASK INPUT JSON\n", 1)[1].split("\n\nFINAL RESPONSE REMINDER", 1)[0]
            )
            payload = {
                "schema_version": "2.0",
                "task_id": "task-a",
                "source_policy_sha256": build_input["source_policy"]["sha256"],
                "feature_summary": "Accept and apply a timeout.",
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
                        "statement": "The client applies the configured timeout.",
                        "category": "public_api",
                        "acceptance_criteria": [
                            {
                                "id": "O01.C01",
                                "statement": "The public client applies the timeout.",
                                "criticality": "must",
                                "basis_kind": "explicit_issue",
                                "source_refs": ["S01"],
                                "derivation": "The issue explicitly requires the timeout.",
                                "open_questions": [],
                                "confidence": "high",
                            }
                        ],
                        "affected_surfaces": ["client.py"],
                        "dependencies": [],
                    }
                ],
            }
        elif role == RoleName.PATCH_REVIEWER:
            assess_input = json.loads(
                prompt.split("TASK INPUT JSON\n", 1)[1].split("\n\nFINAL RESPONSE REMINDER", 1)[0]
            )
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
                            "rationale": "The issue directly requires the behavior.",
                            "compliant_but_rejected_example": None,
                            "satisfies_but_violates_example": None,
                        },
                        "implementation_status": "missing",
                        "implementation_evidence": [
                            {
                                "path": "client.py",
                                "symbol": None,
                                "line_start": 1,
                                "line_end": 1,
                                "explanation": "P0 still uses the wrong timeout.",
                            }
                        ],
                        "test_evidence": [],
                        "validation_note": None,
                        "gap": "The configured timeout is not applied.",
                        "confidence": "high",
                    }
                ],
                "ledger_challenges": [],
                "unexpected_regressions": [],
                "late_obligation_candidates": [],
                "verdict": "request_changes",
                "verdict_rationale": "A source-grounded must criterion is missing.",
                "abstain_reason": None,
                "repair_requests": [
                    {
                        "id": "Q01",
                        "priority": 1,
                        "finding_refs": ["O01.C01"],
                        "criterion_refs": ["O01.C01"],
                        "source_refs": ["S01"],
                        "action_kind": "fix",
                        "problem": "The timeout remains incorrect.",
                        "requested_outcome": "Apply the requested timeout.",
                        "preserve_behaviors": ["Keep the public module importable."],
                        "suggested_validation": ["Read the public value."],
                    }
                ],
            }
        elif role == RoleName.REPAIR:
            repair_input = json.loads(prompt.split("REPAIR INPUT JSON\n", 1)[1])
            Path(worktree, "client.py").write_text("timeout = 20\n")
            Path(worktree, "helper.py").write_text("def timeout_value():\n    return 20\n")
            if self.audit_mode != "missing":
                changed_paths = ["client.py", "helper.py"]
                if self.audit_mode == "wrong-paths":
                    changed_paths = ["client.py"]
                audit = repair_input["audit"]
                audit_payload = {
                    "schema_version": "2.0",
                    "task_id": "task-a",
                    "p0_sha256": repair_input["patch"]["sha256"],
                    "assessment_sha256": audit["assessment_sha256"],
                    "effective_review_set_sha256": audit["effective_review_set_sha256"],
                    "additional_sources": [],
                    "changed_paths": changed_paths,
                    "changes": [
                        {
                            "change_id": "C01",
                            "request_refs": ["Q01"],
                            "criterion_refs": ["O01.C01"],
                            "risk_tags": ["public_api"],
                            "before_behavior": "P0 exposes timeout 10.",
                            "after_behavior": "P1 exposes timeout 20 through public code.",
                            "source_refs": ["S01"],
                            "source_check": "supported",
                            "counterevidence": [],
                            "preserve_behaviors": ["The module remains importable."],
                            "decision": "implement",
                            "changed_paths": changed_paths,
                            "validation": [
                                {
                                    "command": "python -c 'import client'",
                                    "exit_code": 0,
                                    "expected": "The public module imports.",
                                    "actual": "The public module imported.",
                                    "log_path": None,
                                }
                            ],
                        }
                    ],
                    "unaddressed_requests": [],
                    "validation_summary": "The public import probe passed.",
                }
                Path(audit["path"]).write_text(json.dumps(audit_payload))
            payload = None
        else:  # pragma: no cover - this fixture never invokes Solver/Corrector
            raise AssertionError(f"Unexpected role: {role}")

        text = "repair complete" if payload is None else json.dumps(payload)
        return RoleRunResult(
            exit_code=0,
            final_assistant_text=text,
            transcript_path=Path("transcript"),
            trajectory_path=Path("trajectory"),
            started_at="start",
            finished_at="finish",
        )


def make_controller_fixture(tmp_path, *, audit_mode):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init")
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.com")
    (repository / "client.py").write_text("timeout = None\n")
    git(repository, "add", "client.py")
    git(repository, "commit", "-m", "base")
    base = git(repository, "rev-parse", "HEAD").strip()
    (repository / "client.py").write_text("timeout = 10\n")
    p0 = git(repository, "diff", "--binary", "--no-ext-diff", base, "--", ".").encode()
    git(repository, "restore", "client.py")
    p0_path = tmp_path / "p0.patch"
    p0_path.write_bytes(p0)

    workspace = ContainerGitWorkspaceManager(
        environment=LocalEnvironment(),
        transfer_dir=tmp_path / "transfer",
        repository=str(repository),
    )
    workspace.ROOT = str(tmp_path / "worktrees")
    workspace.INPUT_ROOT = str(tmp_path / "contracts")
    roles = GitV2RoleRunner(audit_mode=audit_mode)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = SolveReviewRepairController(
        role_runner=roles,
        workspace=workspace,
        artifacts=artifacts,
        config=WorkflowConfig(
            contract_version="2.0",
            experiment_arm="E2",
            prompt_version="2.2",
            controller_version="2.0.0",
        ),
    )
    task = TaskInput(
        task_id="task-a",
        issue_title="Timeout",
        issue_body="Accept a timeout and apply it to the public client.",
        repository_name="fixture",
        base_commit=base,
    )
    existing = ExistingPatch(
        task_id="task-a",
        base_commit=base,
        path=p0_path,
        sha256=sha256_bytes(p0),
        tracked_diff_sha256=sha256_bytes(p0),
        source_job_id="frozen-p0",
        source_trial_id="task-a__p0",
    )
    return repository, controller, task, existing, artifacts


def test_e2_real_git_audit_uses_p0_to_p1_paths_and_stays_outside_patch(tmp_path):
    repository, controller, task, existing, artifacts = make_controller_fixture(
        tmp_path, audit_mode="valid"
    )

    result = asyncio.run(controller.run(task, existing_p0=existing))

    assert result.state == WorkflowState.COMPLETED
    assert result.repair_audit["changed_paths"] == ["client.py", "helper.py"]
    assert b"client.py" in result.p1_patch
    assert b"helper.py" in result.p1_patch
    assert b"repair_audit.json" not in result.p1_patch
    assert (repository / "client.py").read_text() == "timeout = 20\n"
    assert (repository / "helper.py").is_file()
    assert not (tmp_path / "worktrees" / "repair").exists()
    validation = json.loads((artifacts.root / "repair/repair_audit_validation.json").read_text())
    assert validation["changed_paths"] == ["client.py", "helper.py"]


def test_e2_real_git_patch_reviewer_scratch_is_removed_before_validation(tmp_path):
    class ScratchWritingRunner(GitV2RoleRunner):
        async def run_role(self, **kwargs):
            result = await super().run_role(**kwargs)
            if kwargs["role"] == RoleName.PATCH_REVIEWER:
                Path(kwargs["worktree"], "review-scratch.txt").write_text("temporary\n")
            return result

    repository, controller, task, existing, artifacts = make_controller_fixture(
        tmp_path, audit_mode="valid"
    )
    controller.role_runner = ScratchWritingRunner(audit_mode="valid")

    result = asyncio.run(controller.run(task, existing_p0=existing))

    assert result.state == WorkflowState.COMPLETED
    assert not (repository / "review-scratch.txt").exists()
    events = (artifacts.root / "events.jsonl").read_text()
    assert "PATCH_REVIEWER_UNTRACKED_DISCARDED" in events


def test_v3_repository_source_binding_adds_exact_base_quote_revision_and_digest(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init")
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.com")
    source_bytes = b"first line\nsecond line\nthird line\n"
    (repository / "client.py").write_bytes(source_bytes)
    git(repository, "add", "client.py")
    git(repository, "commit", "-m", "base")
    base = git(repository, "rev-parse", "HEAD").strip()
    (repository / "client.py").write_text("candidate-only line\n")
    manager = ContainerGitWorkspaceManager(
        environment=LocalEnvironment(),
        transfer_dir=tmp_path / "transfer",
        repository=str(repository),
    )

    bound, errors = asyncio.run(
        manager.bind_repository_sources(
            worktree=str(repository),
            base_commit=base,
            sources=[
                {
                    "id": "S01",
                    "kind": "task",
                    "field": "description",
                    "quote": "Read two lines",
                },
                {
                    "id": "S02",
                    "kind": "repository",
                    "path": "client.py",
                    "line_start": 2,
                    "line_end": 3,
                },
            ],
        )
    )

    assert errors == []
    assert bound[0]["quote"] == "Read two lines"
    assert bound[1] == {
        "id": "S02",
        "kind": "repository",
        "path": "client.py",
        "line_start": 2,
        "line_end": 3,
        "quote": "second line\nthird line",
        "base_revision": base,
        "file_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def test_v3_repository_source_binding_rejects_symlink_and_out_of_range_lines(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init")
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.com")
    (repository / "client.py").write_text("one line\n")
    os.symlink("client.py", repository / "client-link.py")
    git(repository, "add", "client.py", "client-link.py")
    git(repository, "commit", "-m", "base")
    base = git(repository, "rev-parse", "HEAD").strip()
    manager = ContainerGitWorkspaceManager(
        environment=LocalEnvironment(),
        transfer_dir=tmp_path / "transfer",
        repository=str(repository),
    )

    bound, errors = asyncio.run(
        manager.bind_repository_sources(
            worktree=str(repository),
            base_commit=base,
            sources=[
                {
                    "id": "S01",
                    "kind": "repository",
                    "path": "client-link.py",
                    "line_start": 1,
                    "line_end": 1,
                },
                {
                    "id": "S02",
                    "kind": "repository",
                    "path": "client.py",
                    "line_start": 1,
                    "line_end": 2,
                },
            ],
        )
    )

    assert bound == []
    assert errors == [
        "S01 repository source is not a regular file: client-link.py",
        "S02 repository source line range is invalid: client.py",
    ]


@pytest.mark.parametrize("audit_mode", ["missing", "wrong-paths"])
def test_e2_real_git_invalid_audit_discards_candidate_and_publishes_p0(tmp_path, audit_mode):
    repository, controller, task, existing, artifacts = make_controller_fixture(
        tmp_path, audit_mode=audit_mode
    )

    result = asyncio.run(controller.run(task, existing_p0=existing))

    assert result.state == WorkflowState.REPAIR_AUDIT_FAILED
    assert result.p1_patch == existing.path.read_bytes()
    assert (repository / "client.py").read_text() == "timeout = 10\n"
    assert not (repository / "helper.py").exists()
    assert (artifacts.root / "repair/discarded_candidate.patch").is_file()
    manifest = json.loads((artifacts.root / "manifest.json").read_text())
    assert manifest["repair_audit_status"] == "failed"
