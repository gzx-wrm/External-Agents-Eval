from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from external_agents_eval.workflows.solve_review_repair.artifacts import (
    ArtifactStore,
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.models import (
    RoleName,
    RoleRunResult,
    TaskInput,
)
from external_agents_eval.workflows.solve_review_repair.v6_config import V6Config
from external_agents_eval.workflows.solve_review_repair.v6_controller import V6Controller
from external_agents_eval.workflows.solve_review_repair.v6_fork import (
    V6ForkPlan,
    V6ForkPlans,
    V6StageSource,
    write_stage,
)
from external_agents_eval.workflows.solve_review_repair.v6_prompts import solver_prompt


class _SpecRunner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[RoleName] = []

    async def run_role(self, *, role: RoleName, **_kwargs) -> RoleRunResult:
        self.calls.append(role)
        trajectory = self.root / f"{len(self.calls)}.json"
        trajectory.write_text("{}", encoding="utf-8")
        payload = {
            "summary": "Implement retries without changing terminal failures.",
            "obligations": [
                {
                    "statement": "Retry transient transport failures.",
                    "importance": "required",
                    "sources": [
                        {
                            "kind": "task",
                            "field": "description",
                            "quote": "Retry transient transport failures.",
                        }
                    ],
                }
            ],
            "preservation_obligations": [
                {
                    "statement": "Terminal failures remain observable.",
                    "importance": "required",
                    "sources": [
                        {
                            "kind": "task",
                            "field": "description",
                            "quote": "Terminal failures remain observable.",
                        }
                    ],
                }
            ],
            "open_questions": [
                {
                    "question": "Which failures are transient?",
                    "blocking": False,
                    "sources": [
                        {
                            "kind": "task",
                            "field": "description",
                            "quote": "transient transport failures",
                        }
                    ],
                }
            ],
        }
        return RoleRunResult(
            exit_code=0,
            final_assistant_text=json.dumps(payload),
            transcript_path=trajectory,
            trajectory_path=trajectory,
            started_at="now",
            finished_at="now",
        )


class _BaseWorkspace:
    async def tracked_diff_digest(self, **_kwargs) -> str:
        return "frozen-base"


class _DegradedWorkspace(_BaseWorkspace):
    async def bind_repository_sources(self, **_kwargs):
        return [], ["repository source line range is outside the frozen file"]


class _InvalidSpecRunner(_SpecRunner):
    async def run_role(self, *, role: RoleName, **_kwargs) -> RoleRunResult:
        result = await super().run_role(role=role)
        return RoleRunResult(
            exit_code=result.exit_code,
            final_assistant_text="{}",
            transcript_path=result.transcript_path,
            trajectory_path=result.trajectory_path,
            started_at=result.started_at,
            finished_at=result.finished_at,
        )


def _task() -> TaskInput:
    return TaskInput(
        task_id="retry-task",
        issue_body=(
            "Retry transient transport failures. "
            "Terminal failures remain observable."
        ),
        repository_name="repo",
        base_commit="base",
        issue_title="Retry requests",
    )


def test_v6_prepares_one_frozen_ledger_and_reuses_it(tmp_path):
    runner = _SpecRunner(tmp_path)
    controller = V6Controller(
        role_runner=runner,
        workspace=_BaseWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    try:
        first = asyncio.run(
            controller.prepare_ledger(
                _task(),
                base_worktree="/base",
                stage_id="pre-solver",
            )
        )
        second = asyncio.run(
            controller.prepare_ledger(
                _task(),
                base_worktree="/base",
                stage_id="pre-solver",
            )
        )
    finally:
        controller.close()

    assert first == second
    assert runner.calls == [RoleName.V5_SPECIFICATION_REVIEWER]
    assert [item["id"] for item in first["obligations"]] == ["O001"]
    assert [item["id"] for item in first["preservation_obligations"]] == ["P001"]
    assert [item["id"] for item in first["open_questions"]] == ["Q001"]
    assert (tmp_path / "artifacts/v5/ledger.json").is_file()
    envelope = json.loads(
        (tmp_path / "artifacts/v6/pre-solver-ledger.json").read_text()
    )
    assert envelope["ledger"] == first
    assert envelope["ledger_sha256"]


def test_v6_solver_prompt_preserves_original_task_and_complete_ledger():
    ledger = {
        "summary": "Implement and preserve behavior.",
        "obligations": [{"id": "O001", "content": {"sources": []}}],
        "preservation_obligations": [
            {"id": "P001", "content": {"sources": []}}
        ],
        "open_questions": [{"id": "Q001", "content": {"sources": []}}],
    }

    prompt = solver_prompt("Original coding instruction.", ledger)

    assert prompt.startswith("Original coding instruction.\n")
    assert all(identifier in prompt for identifier in ("O001", "P001", "Q001"))
    assert "original task and applicable public contracts remain authoritative" in prompt
    assert "Do not edit\nor regenerate the ledger" in prompt


def test_v6_degraded_ledger_is_explicitly_non_authoritative_to_solver():
    prompt = solver_prompt(
        "Original coding instruction.",
        {"summary": "Use the task.", "obligations": [], "preservation_obligations": [], "open_questions": []},
        ledger_quality="degraded",
        ledger_warnings=["line range is outside the frozen file"],
    )

    assert "LEDGER QUALITY" in prompt
    assert "mechanical source-anchor" in prompt
    assert "line range is outside the frozen file" in prompt
    assert "inspect the\nbase repository directly" in prompt


def test_v6_unparseable_specification_terminates_before_publishing_ledger(tmp_path):
    runner = _InvalidSpecRunner(tmp_path)
    controller = V6Controller(
        role_runner=runner,
        workspace=_BaseWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        config=V6Config(max_format_retries=0, max_semantic_retries=0),
    )
    try:
        ledger = asyncio.run(
            controller.prepare_ledger(
                _task(),
                base_worktree="/base",
                stage_id="invalid-pre-solver",
            )
        )
    finally:
        controller.close()

    assert runner.calls == [RoleName.V5_SPECIFICATION_REVIEWER]
    assert ledger is None
    assert controller.ledger_quality == "unavailable"
    assert not (tmp_path / "artifacts/v5/ledger.json").exists()
    envelope = json.loads(
        (tmp_path / "artifacts/v6/pre-solver-ledger.json").read_text()
    )
    assert envelope["ledger_quality"] == "unavailable"
    assert envelope["ledger"] is None


def test_v6_source_binding_failure_is_degraded_and_non_blocking(tmp_path):
    class _RepositorySpecRunner(_SpecRunner):
        async def run_role(self, *, role: RoleName, **kwargs) -> RoleRunResult:
            result = await super().run_role(role=role, **kwargs)
            payload = json.loads(result.final_assistant_text)
            payload["obligations"][0]["sources"] = [
                {
                    "kind": "repository",
                    "path": "src/retry.py",
                    "line_start": 999,
                    "line_end": 1000,
                }
            ]
            return RoleRunResult(
                exit_code=result.exit_code,
                final_assistant_text=json.dumps(payload),
                transcript_path=result.transcript_path,
                trajectory_path=result.trajectory_path,
                started_at=result.started_at,
                finished_at=result.finished_at,
            )

    runner = _RepositorySpecRunner(tmp_path)
    controller = V6Controller(
        role_runner=runner,
        workspace=_DegradedWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        config=V6Config(max_format_retries=0, max_semantic_retries=0),
    )
    try:
        ledger = asyncio.run(
            controller.prepare_ledger(
                _task(),
                base_worktree="/base",
                stage_id="degraded-pre-solver",
            )
        )
        second = asyncio.run(
            controller.prepare_ledger(
                _task(),
                base_worktree="/base",
                stage_id="degraded-pre-solver",
            )
        )
    finally:
        controller.close()

    assert ledger is not None
    assert controller.ledger_quality == "degraded"
    assert controller.ledger_warnings
    assert (tmp_path / "artifacts/v5/ledger.json").is_file()
    envelope = json.loads(
        (tmp_path / "artifacts/v6/pre-solver-ledger.json").read_text()
    )
    assert envelope["ledger_quality"] == "degraded"
    assert envelope["ledger"] == ledger
    assert second == ledger
    assert controller.ledger_quality == "degraded"


def test_v6_fork_plan_selects_explicit_predecessors(tmp_path):
    path = tmp_path / "fork.json"
    path.write_text(json.dumps({
        "task_id": "retry-task",
        "start_at": "I",
        "source_run": "original",
        "reuse": {"R": "better-regression"},
    }), encoding="utf-8")
    plan = V6ForkPlan.load(path)
    assert plan.sources == {
        "A": (tmp_path / "original").resolve(),
        "Solver": (tmp_path / "original").resolve(),
        "R": (tmp_path / "better-regression").resolve(),
    }
    assert plan.start_at == "I"


def test_v6_fork_plans_route_each_task_to_its_own_ledger(tmp_path):
    from external_agents_eval.agents.pier.pi_v6_solve_review_repair import (
        PiV6SolveReviewRepairPierAgent,
    )

    mapping = {}
    for task_id in ("task-a", "task-b"):
        plan_path = tmp_path / task_id / "fork.json"
        plan_path.parent.mkdir()
        plan_path.write_text(json.dumps({
            "task_id": task_id, "start_at": "Solver", "source_run": "ledger.json",
        }), encoding="utf-8")
        mapping[task_id] = f"{task_id}/fork.json"
    path = tmp_path / "plans.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    plans = V6ForkPlans.load(path)
    agent = object.__new__(PiV6SolveReviewRepairPierAgent)
    agent.fork_plan = None
    agent.fork_plans = plans
    for task_id in mapping:
        selected_id, plan = agent._fork_for_task(task_id)
        assert selected_id == task_id
        assert plan.source("A") == (tmp_path / task_id / "ledger.json").resolve()

    long_id = "sqlite-utils-safe-import-checkpoints"
    long_plan = tmp_path / "sqlite.json"
    long_plan.write_text(json.dumps({"task_id": long_id, "start_at": "A"}), encoding="utf-8")
    mapping[long_id] = "sqlite.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    agent.fork_plans = V6ForkPlans.load(path)
    selected_id, plan = agent._fork_for_task(long_id[:32].rstrip("_-"))
    assert selected_id == plan.task_id == long_id

    (tmp_path / "task-b/fork.json").write_text(json.dumps({
        "task_id": "task-a", "start_at": "Solver", "source_run": "ledger.json",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match mapping key"):
        V6ForkPlans.load(path)


def test_v6_imported_a_skips_model_and_binds_sources(tmp_path):
    original = V6Controller(
        role_runner=_SpecRunner(tmp_path),
        workspace=_BaseWorkspace(),
        artifacts=ArtifactStore(tmp_path / "original"),
    )
    try:
        ledger = asyncio.run(original.prepare_ledger(_task(), base_worktree="/base"))
    finally:
        original.close()
    assert ledger is not None
    runner = _SpecRunner(tmp_path)
    imported = V6Controller(
        role_runner=runner,
        workspace=_BaseWorkspace(),
        artifacts=ArtifactStore(tmp_path / "imported"),
    )
    source = V6StageSource(
        tmp_path / "original", task_id="retry-task",
        base_commit="base", baseline_sha256=sha256_bytes(b""),
    )
    try:
        reused = asyncio.run(imported.prepare_ledger(
            _task(), base_worktree="/base", imported_ledger=source.ledger(),
        ))
    finally:
        imported.close()
    assert reused == ledger
    assert runner.calls == []


def test_v6_stage_source_checks_solver_ledger_and_patch(tmp_path):
    root = tmp_path / "source"
    (root / "outputs").mkdir(parents=True)
    (root / "outputs/v6-p0.patch").write_bytes(b"candidate")
    write_stage(
        root, stage="Solver", task_id="retry-task", base_commit="base",
        baseline_sha256=sha256_bytes(b""), ledger_sha256=sha256_json({"x": 1}),
        patch_sha256=sha256_bytes(b"candidate"),
    )
    source = V6StageSource(
        root, task_id="retry-task", base_commit="base",
        baseline_sha256=sha256_bytes(b""),
    )
    assert source.patch("Solver", ledger_sha256=sha256_json({"x": 1})) == b"candidate"
    with pytest.raises(ValueError, match="another A ledger"):
        source.patch("Solver", ledger_sha256=sha256_json({"x": 2}))
    (root / "outputs/v6-p0.patch").write_bytes(b"changed")
    with pytest.raises(ValueError, match="has changed"):
        source.patch("Solver", ledger_sha256=sha256_json({"x": 1}))


def test_v6_stage_source_reads_historical_completed_run_without_base_field(tmp_path):
    root = tmp_path / "historical"
    full_task_id = "sqlite-utils-safe-import-checkpoints"
    historical_task_id = full_task_id[:32].rstrip("_-")
    (root / "v6").mkdir(parents=True)
    (root / "outputs").mkdir()
    ledger = {"obligations": [], "preservation_obligations": [], "open_questions": []}
    baseline_sha = sha256_bytes(b"")
    (root / "v6/pre-solver-ledger.json").write_text(json.dumps({
        "task_id": historical_task_id, "base_commit": "base",
        "baseline_patch_sha256": baseline_sha,
        "ledger": ledger, "ledger_sha256": sha256_json(ledger),
    }), encoding="utf-8")
    (root / "outputs/v6-p0.patch").write_bytes(b"p0")
    (root / "outputs/v5-p1.patch").write_bytes(b"p1")
    (root / "manifest.json").write_text(json.dumps({
        "baseline_patch_sha256": baseline_sha,
        "ledger_sha256": sha256_json(ledger),
        "p0_sha256": sha256_bytes(b"p0"),
        "p1_sha256": sha256_bytes(b"p1"),
        "route": "repair",
    }), encoding="utf-8")
    source = V6StageSource(
        root, task_id=full_task_id, base_commit="base", baseline_sha256=baseline_sha,
    )
    assert source.patch("Solver", ledger_sha256=sha256_json(ledger)) == b"p0"
    assert source.patch("P1") == b"p1"
    assert source.p1_route() == "repair"
