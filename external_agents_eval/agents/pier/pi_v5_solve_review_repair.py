from __future__ import annotations

import json
import re
import shlex
import uuid
from pathlib import Path
from typing import Any

from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext

from external_agents_eval.agents.pier.pi import PiPierAgent
from external_agents_eval.agents.pier.pi_role_runner import (
    ContainerGitWorkspaceManager,
    PiRoleRunner,
)
from external_agents_eval.agents.pier.pi_solve_review_repair import (
    issue_title_from_instruction,
    resolve_frozen_task_id,
)
from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    load_reviewer_experiment_manifest,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore, sha256_bytes
from external_agents_eval.workflows.solve_review_repair.models import RoleName, TaskInput
from external_agents_eval.workflows.solve_review_repair.v5_config import V5Config
from external_agents_eval.workflows.solve_review_repair.v5_controller import V5Controller

_DEEPSWE_MANAGED_DELIVERY = (
    "Please work on this in a new branch from main and commit everything when you are done.",
)


def managed_delivery_from_instruction(instruction: str) -> tuple[dict[str, Any], ...]:
    """Bind only the benchmark delivery boilerplate that this adapter owns."""

    return tuple(
        {"kind": "task", "field": "description", "quote": quote}
        for quote in _DEEPSWE_MANAGED_DELIVERY
        if quote in instruction
    )


class PiV5SolveReviewRepairPierAgent(PiPierAgent):
    """Pier adapter for the first executable V5 A→I/R→S→Repair pipeline."""

    def __init__(self, *args: Any, workflow_timeout_sec: int | None = None, p0_manifest: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.workflow_timeout_sec = workflow_timeout_sec
        self.p0_manifest = Path(p0_manifest).expanduser().resolve() if p0_manifest else None

    @staticmethod
    def name() -> str:
        return "pi-v5-review-repair"

    async def _resolve_commit(
        self,
        environment: BaseEnvironment,
        revision: str,
    ) -> str:
        """Resolve one trusted revision to an unambiguous commit object."""
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}|HEAD", revision):
            raise RuntimeError("V5 received an invalid Git revision")
        expression = revision if revision == "HEAD" else f"{revision}^{{commit}}"
        result = await self.checked_exec(
            environment,
            f"git rev-parse --verify {shlex.quote(expression)}",
            cwd=self.workdir,
        )
        resolved = (result.stdout or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", resolved):
            raise RuntimeError("V5 could not resolve the Git revision to a commit")
        return resolved

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        task_id = environment.session_id.rsplit("__", 1)[0]
        base_commit = await self._resolve_commit(environment, "HEAD")
        existing_p0 = None
        if self.p0_manifest is not None:
            manifest = load_reviewer_experiment_manifest(self.p0_manifest)
            task_id = resolve_frozen_task_id(task_id, manifest.task_ids)
            existing_p0 = manifest.patches[task_id]
            if sha256_bytes(existing_p0.path.read_bytes()) != existing_p0.sha256:
                raise RuntimeError("V5 P0 manifest digest does not match its patch")
            manifest_base = await self._resolve_commit(
                environment,
                existing_p0.base_commit,
            )
            if manifest_base != base_commit:
                raise RuntimeError("V5 P0 manifest base commit does not match the task")
        root = self.logs_dir.parent / "v5_review_repair"
        store = ArtifactStore(
            root,
            secrets=tuple(
                self.extra_env.get(key, "")
                for key in self.stage_model_config.required_env_keys()
            ),
        )
        env = self.agent_env(environment)
        env.update({
            "PI_CODING_AGENT_DIR": "/external_eval/pi_config",
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
            "HOME": "/external_eval/home",
            "PATH": "/opt/pi-runtime/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        })
        runner = PiRoleRunner(
            environment=environment, logs_root=root, model=self.runtime_model,
            provider=self.runtime_provider,
            stage_model_config=self.stage_model_config,
            agent_version=self.version(), env=env, redact=self.redact,
        )
        workspace = ContainerGitWorkspaceManager(
            environment=environment, transfer_dir=root / ".transfer",
            repository=self.workdir, redact=self.redact,
        )
        controller: V5Controller | None = None
        try:
            # The task image may start with an environment-only diff (for
            # example a dependency compatibility edit).  Record it for
            # diagnosis, then normalize /app before delivery so reviewer
            # scratch cannot leak into the published patch.
            environment_patch = await workspace.capture_patch(
                worktree=self.workdir, base_commit=base_commit
            )
            environment_baseline_sha256 = sha256_bytes(environment_patch)
            baseline_manifest = None
            baseline_manifest_method = getattr(workspace, "baseline_manifest", None)
            if baseline_manifest_method is not None:
                baseline_manifest = await baseline_manifest_method(
                    worktree=self.workdir,
                    base_commit=base_commit,
                    patch=environment_patch,
                )
            store.write_bytes("inputs/environment.patch", environment_patch)
            if baseline_manifest is not None:
                store.write_json("inputs/environment-baseline.json", baseline_manifest)
            store.event(
                "V5_FINAL_WORKTREE_BASELINE",
                patch_sha256=environment_baseline_sha256,
                had_changes=bool(environment_patch),
                baseline_manifest=baseline_manifest,
            )
            paths = await workspace.prepare(base_commit=base_commit)
            # Every role must observe the same runnable task image.  The task
            # Dockerfile may leave a deterministic, tracked baseline change
            # in /app (numba's llvmlite compatibility pin is one example),
            # while Git worktrees are created from the clean base commit.
            # Materialize that baseline into each disposable role worktree
            # before any agent starts.
            baseline_worktrees = [
                paths.solver,
                paths.spec_review,
                paths.patch_review,
                paths.repair,
                paths.summary,
            ]
            for worktree in dict.fromkeys(path for path in baseline_worktrees if path):
                if environment_patch:
                    await workspace.apply_patch(worktree=worktree, patch=environment_patch)
            if existing_p0 is None:
                solve = await runner.run_role(
                    role=RoleName.SOLVER, prompt=instruction, worktree=paths.solver,
                    tools=("read", "bash", "edit", "write", "grep", "find", "ls"),
                    timeout_sec=self.workflow_timeout_sec or 10_800,
                    artifact_prefix="v5/solver",
                )
                if solve.exit_code != 0 or solve.timed_out:
                    raise RuntimeError(f"V5 solver failed: {solve.exception or solve.exit_code}")
                p0 = await workspace.capture_patch(
                    worktree=paths.solver,
                    base_commit=base_commit,
                    baseline_patch=environment_patch,
                )
            else:
                # Historical P0 manifests contain a full base-relative patch.
                # Normalize them once into the new baseline-relative coordinate
                # system so old reviewer runs remain usable after all role
                # worktrees start from base_commit + baseline.
                raw_p0 = existing_p0.path.read_bytes()
                await workspace.restore_reviewer_worktree(
                    worktree=paths.patch_review,
                    base_commit=base_commit,
                )
                await workspace.apply_patch(
                    worktree=paths.patch_review,
                    patch=raw_p0,
                )
                applied_digest = await workspace.tracked_diff_digest(
                    worktree=paths.patch_review,
                    base_commit=base_commit,
                )
                if applied_digest != existing_p0.tracked_diff_sha256:
                    raise RuntimeError("V5 applied P0 differs from the frozen tracked diff")
                p0 = await workspace.capture_patch(
                    worktree=paths.patch_review,
                    base_commit=base_commit,
                    baseline_patch=environment_patch,
                )
                await workspace.restore_reviewer_worktree(
                    worktree=paths.patch_review,
                    base_commit=base_commit,
                    baseline_patch=environment_patch,
                )
            regression_worktree = await workspace.prepare_additional_worktree(
                name="regression-review",
                base_commit=base_commit,
            )
            if environment_patch:
                await workspace.apply_patch(worktree=regression_worktree, patch=environment_patch)
            await workspace.apply_patch(worktree=paths.patch_review, patch=p0)
            await workspace.apply_patch(worktree=regression_worktree, patch=p0)
            summary_worktree = paths.summary or paths.patch_review
            if summary_worktree != paths.patch_review:
                await workspace.apply_patch(worktree=summary_worktree, patch=p0)
            task = TaskInput(
                task_id=task_id,
                issue_title=issue_title_from_instruction(instruction, task_id),
                issue_body=instruction,
                repository_name=task_id.split("-", 1)[0],
                base_commit=base_commit,
                managed_delivery=managed_delivery_from_instruction(instruction),
            )
            config = V5Config(
                role_timeout_sec=self.workflow_timeout_sec or 1_800,
                model=self.model_spec_for_role("solver").model,
            )
            controller = V5Controller(
                role_runner=runner,
                workspace=workspace,
                artifacts=store,
                config=config,
            )
            try:
                result = await controller.run(
                    task,
                    candidate_patch=p0,
                    baseline_patch=environment_patch,
                    base_worktree=paths.spec_review,
                    candidate_worktree=paths.patch_review,
                    summary_worktree=summary_worktree,
                    regression_worktree=regression_worktree,
                    repair_worktree=paths.repair,
                    run_id=uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{task_id}:{base_commit}:{sha256_bytes(p0)}:"
                        f"{environment_baseline_sha256}:"
                        f"{self.stage_model_config.digest}:"
                        f"{config.prompt_version}:{config.controller_version}",
                    ).hex,
                )
                final_patch = result.p1_patch
                metadata = {
                    **self.model_spec_for_role("solver").to_metadata(),
                    "stage_models": self.stage_model_config.to_metadata(),
                    "workflow": "solve-review-repair-v5",
                    "workflow_state": "completed",
                    "route": result.route,
                    "p0_sha256": result.p0_sha256,
                    "p1_sha256": sha256_bytes(result.p1_patch),
                    "patch_base": "environment-baseline" if environment_patch else "base-commit",
                    "freeze_id": result.freeze_id,
                    "repair_executed": result.route == "repair",
                    "environment_baseline_sha256": environment_baseline_sha256,
                    "environment_baseline": baseline_manifest,
                    "prompt_version": config.prompt_version,
                    "controller_version": config.controller_version,
                }
            except Exception as error:  # noqa: BLE001 - any workflow failure safely falls back to P0
                # V5 treats invalid review protocol output as a safe hold.  P0
                # remains the candidate submitted to the independent evaluator,
                # while the full error and partial artifacts stay available for
                # experiment analysis.
                store.event("V5_SAFE_FALLBACK", error=str(error), task_id=task_id)
                final_patch = p0
                metadata = {
                    **self.model_spec_for_role("solver").to_metadata(),
                    "stage_models": self.stage_model_config.to_metadata(),
                    "workflow": "solve-review-repair-v5",
                    "workflow_state": "safe_fallback",
                    "route": "hold",
                    "workflow_error": self.redact(str(error)),
                    "p0_sha256": sha256_bytes(p0),
                    "p1_sha256": sha256_bytes(p0),
                    "patch_base": "environment-baseline" if environment_patch else "base-commit",
                    "freeze_id": None,
                    "repair_executed": None,
                    "environment_baseline_sha256": environment_baseline_sha256,
                    "environment_baseline": baseline_manifest,
                    "prompt_version": config.prompt_version,
                    "controller_version": config.controller_version,
                }
            store.write_json(
                "manifest.json",
                {
                    "schema_version": "v5-run-1",
                    "final_state": "COMPLETED",
                    "workflow_state": metadata["workflow_state"],
                    "route": metadata["route"],
                    "repair_executed": metadata["repair_executed"],
                    "contract_version": "5.0",
                    "prompt_version": metadata["prompt_version"],
                    "controller_version": metadata["controller_version"],
                    "p0_sha256": metadata["p0_sha256"],
                    "p1_sha256": metadata["p1_sha256"],
                    "patch_base": metadata["patch_base"],
                    "baseline_patch_sha256": environment_baseline_sha256,
                    "freeze_id": metadata["freeze_id"],
                    "environment_baseline_sha256": metadata[
                        "environment_baseline_sha256"
                    ],
                    "environment_baseline": metadata.get("environment_baseline"),
                    "stage_models": metadata.get("stage_models"),
                },
            )
            discarded_final = await workspace.restore_final_worktree(base_commit=base_commit)
            if discarded_final:
                store.write_bytes(
                    "outputs/discarded_final_worktree.patch", discarded_final
                )
            if environment_patch:
                await workspace.apply_patch(
                    worktree=self.workdir, patch=environment_patch
                )
            store.event(
                "V5_FINAL_WORKTREE_NORMALIZED",
                discarded_changes=bool(discarded_final),
                discarded_patch_sha256=sha256_bytes(discarded_final),
                baseline_restored=bool(environment_patch),
                baseline_patch_sha256=environment_baseline_sha256,
            )
            await workspace.publish_final_patch(
                base_commit=base_commit,
                patch=final_patch,
                baseline_patch=environment_patch,
                baseline_manifest=baseline_manifest,
            )
            await self.commit_solution(environment)
            self.save_trajectory(self._aggregate_trajectories(instruction, root), context)
            context.metadata = metadata
        finally:
            if controller is not None:
                controller.close()
            await workspace.cleanup()

    def _aggregate_trajectories(self, instruction: str, root: Path) -> dict[str, Any]:
        steps: list[dict[str, Any]] = [
            {"step_id": 1, "source": "user", "message": instruction}
        ]
        totals = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
        }
        seen_metric = {key: False for key in totals}
        # Evidence bundles archive copies of reviewer trajectories. Count the
        # original invocation once, not its archived copy as a second call.
        for path in sorted(root.glob("**/trajectory.json")):
            if "bundles" in path.relative_to(root).parts:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for step in payload.get("steps", [])[1:]:
                copied = dict(step)
                copied["step_id"] = len(steps) + 1
                steps.append(copied)
            metrics = payload.get("final_metrics") or {}
            for key in totals:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += value
                    seen_metric[key] = True
        final_metrics: dict[str, Any] = {
            "total_steps": len(steps),
            "extra": {"agent_steps": sum(step["source"] == "agent" for step in steps)},
        }
        final_metrics.update(
            {key: value for key, value in totals.items() if seen_metric[key]}
        )
        return {
            "schema_version": "ATIF-v1.7",
            "session_id": str(uuid.uuid4()),
            "trajectory_id": str(uuid.uuid4()),
            "agent": {
                "name": self.name(),
                "version": self.version(),
                "model_name": self.runtime_model,
            },
            "steps": steps,
            "final_metrics": final_metrics,
        }
