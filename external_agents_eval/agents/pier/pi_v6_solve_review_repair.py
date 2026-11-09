from __future__ import annotations

import uuid
from typing import Any

from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext

from external_agents_eval.agents.pier.pi_role_runner import (
    ContainerGitWorkspaceManager,
    PiRoleRunner,
)
from external_agents_eval.agents.pier.pi_solve_review_repair import (
    issue_title_from_instruction,
    resolve_frozen_task_id,
)
from external_agents_eval.agents.pier.pi_v5_solve_review_repair import (
    PiV5SolveReviewRepairPierAgent,
    managed_delivery_from_instruction,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import (
    ArtifactStore,
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.models import RoleName, TaskInput
from external_agents_eval.workflows.solve_review_repair.v6_config import (
    V6Config,
)
from external_agents_eval.workflows.solve_review_repair.v6_controller import V6Controller
from external_agents_eval.workflows.solve_review_repair.v6_fork import (
    V6ForkPlan,
    V6ForkPlans,
    V6StageSource,
    write_stage,
)
from external_agents_eval.workflows.solve_review_repair.v6_prompts import solver_prompt


class PiV6SolveReviewRepairPierAgent(PiV5SolveReviewRepairPierAgent):
    """Run A before Solver, then reuse the frozen ledger in V5 review."""

    workflow_version = "6.1"
    workflow_config_type = V6Config

    def _make_controller(self, **kwargs: Any) -> V6Controller:
        return V6Controller(**kwargs)

    def _make_role_runner(self, **kwargs: Any) -> PiRoleRunner:
        return PiRoleRunner(**kwargs)

    def _make_workspace(self, **kwargs: Any) -> ContainerGitWorkspaceManager:
        return ContainerGitWorkspaceManager(**kwargs)

    def __init__(
        self,
        *args: Any,
        workflow_timeout_sec: int | None = None,
        p0_manifest: str | None = None,
        fork_plan: str | None = None,
        fork_plans: str | None = None,
        **kwargs: Any,
    ) -> None:
        if p0_manifest is not None:
            raise ValueError("V6 generates P0 and does not accept p0_manifest")
        if fork_plan and fork_plans:
            raise ValueError("Use either fork_plan or fork_plans")
        super().__init__(
            *args,
            workflow_timeout_sec=workflow_timeout_sec,
            p0_manifest=None,
            **kwargs,
        )
        self.fork_plan = V6ForkPlan.load(fork_plan) if fork_plan else None
        self.fork_plans = V6ForkPlans.load(fork_plans) if fork_plans else None

    @staticmethod
    def name() -> str:
        return "pi-v6-review-repair"

    def _fork_for_task(self, session_task_id: str) -> tuple[str, V6ForkPlan | None]:
        if self.fork_plans is not None:
            task_id = resolve_frozen_task_id(session_task_id, tuple(self.fork_plans.plans))
            return task_id, self.fork_plans.plans[task_id]
        if self.fork_plan is not None:
            task_id = resolve_frozen_task_id(session_task_id, (self.fork_plan.task_id,))
            return task_id, self.fork_plan
        return session_task_id, None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        task_id, fork = self._fork_for_task(environment.session_id.rsplit("__", 1)[0])
        base_commit = await self._resolve_commit(environment, "HEAD")
        root = self.logs_dir.parent / "v6_review_repair"
        store = ArtifactStore(
            root,
            secrets=tuple(
                self.extra_env.get(key, "")
                for key in self.stage_model_config.required_env_keys()
            ),
        )
        if fork is not None:
            store.write_json(
                "inputs/fork-plan.json",
                {
                    "task_id": fork.task_id,
                    "start_at": fork.start_at,
                    "reuse": {stage: str(path) for stage, path in fork.sources.items()},
                    "plan_sha256": fork.digest,
                },
            )
        env = self.agent_env(environment)
        env.update(
            {
                "PI_CODING_AGENT_DIR": "/external_eval/pi_config",
                "PI_OFFLINE": "1",
                "PI_SKIP_VERSION_CHECK": "1",
                "PI_TELEMETRY": "0",
                "HOME": "/external_eval/home",
                "PATH": (
                    "/opt/pi-runtime/bin:/usr/local/sbin:/usr/local/bin:"
                    "/usr/sbin:/usr/bin:/sbin:/bin"
                ),
            }
        )
        runner = self._make_role_runner(
            environment=environment,
            logs_root=root,
            model=self.runtime_model,
            provider=self.runtime_provider,
            stage_model_config=self.stage_model_config,
            agent_version=self.version(),
            env=env,
            redact=self.redact,
        )
        workspace = self._make_workspace(
            environment=environment,
            transfer_dir=root / ".transfer",
            repository=self.workdir,
            redact=self.redact,
        )
        controller: V6Controller | None = None
        try:
            environment_patch = await workspace.capture_patch(
                worktree=self.workdir,
                base_commit=base_commit,
            )
            environment_baseline_sha256 = sha256_bytes(environment_patch)
            def source(stage: str) -> V6StageSource:
                assert fork is not None
                return V6StageSource(
                    fork.source(stage),
                    task_id=task_id,
                    base_commit=base_commit,
                    baseline_sha256=environment_baseline_sha256,
                )
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
                "V6_FINAL_WORKTREE_BASELINE",
                patch_sha256=environment_baseline_sha256,
                had_changes=bool(environment_patch),
                baseline_manifest=baseline_manifest,
            )

            paths = await workspace.prepare(base_commit=base_commit)
            baseline_worktrees = [
                paths.solver,
                paths.spec_review,
                paths.patch_review,
                paths.repair,
                paths.summary,
            ]
            for worktree in dict.fromkeys(
                path for path in baseline_worktrees if path
            ):
                if environment_patch:
                    await workspace.apply_patch(
                        worktree=worktree,
                        patch=environment_patch,
                    )

            task = TaskInput(
                task_id=task_id,
                issue_title=issue_title_from_instruction(instruction, task_id),
                issue_body=instruction,
                repository_name=task_id.split("-", 1)[0],
                base_commit=base_commit,
                managed_delivery=managed_delivery_from_instruction(instruction),
            )
            config = self.workflow_config_type(
                role_timeout_sec=self.workflow_timeout_sec or 1_800,
                model=self.model_spec_for_role("solver").model,
            )
            controller = self._make_controller(
                role_runner=runner,
                workspace=workspace,
                artifacts=store,
                config=config,
            )
            if fork is not None and fork.start_at == "Verifier":
                p1_source = source("P1")
                final_patch = p1_source.patch("P1")
                route = p1_source.p1_route()
                store.write_bytes("outputs/v5-p1.patch", final_patch)
                write_stage(
                    root,
                    stage="P1",
                    task_id=task_id,
                    base_commit=base_commit,
                    baseline_sha256=environment_baseline_sha256,
                    patch_sha256=sha256_bytes(final_patch),
                    route=route,
                    reused_from=str(p1_source.root),
                )
                store.write_json("manifest.json", {
                    "schema_version": f"v{self.workflow_version}-run-1",
                    "task_id": task_id,
                    "base_commit": base_commit,
                    "baseline_patch_sha256": environment_baseline_sha256,
                    "final_state": "COMPLETED",
                    "workflow_state": "verifier_only",
                    "route": route,
                    "p1_sha256": sha256_bytes(final_patch),
                    "fork_plan_sha256": fork.digest,
                })
                await self._publish_v6_patch(
                    workspace=workspace,
                    store=store,
                    base_commit=base_commit,
                    environment_patch=environment_patch,
                    baseline_manifest=baseline_manifest,
                    final_patch=final_patch,
                    verify_exact=True,
                )
                await self.commit_solution(environment)
                self.save_trajectory(self._aggregate_trajectories(instruction, root), context)
                context.metadata = {
                    **self.model_spec_for_role("solver").to_metadata(),
                    "workflow": f"solve-review-repair-v{self.workflow_version}",
                    "workflow_state": "verifier_only",
                    "route": route,
                    "p1_sha256": sha256_bytes(final_patch),
                    "fork_plan_sha256": fork.digest,
                }
                return
            specification_stage_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"v{self.workflow_version}-spec:{task_id}:{base_commit}:"
                f"{sha256_bytes(instruction.encode('utf-8'))}:"
                f"{environment_baseline_sha256}:"
                f"{self.stage_model_config.digest}:"
                f"{config.workflow_prompt_version}:"
                f"{config.workflow_controller_version}:"
                f"{config.prompt_version}:{config.controller_version}"
                f"{':' + fork.digest if fork is not None else ''}",
            ).hex
            ledger = await controller.prepare_ledger(
                task,
                base_worktree=paths.spec_review,
                baseline_patch=environment_patch,
                stage_id=specification_stage_id,
                imported_ledger=(
                    source("A").ledger() if fork is not None and "A" in fork.sources else None
                ),
            )

            if ledger is None:
                metadata = {
                    **self.model_spec_for_role("solver").to_metadata(),
                    "stage_models": self.stage_model_config.to_metadata(),
                    "workflow": f"solve-review-repair-v{self.workflow_version}",
                    "workflow_state": "specification_unavailable",
                    "route": "terminated_before_p0",
                    "final_state": "LEDGER_SCHEMA_FAILED",
                    "p0_generated": False,
                    "p1_generated": False,
                    "ledger_quality": controller.ledger_quality,
                    "ledger_warnings": list(controller.ledger_warnings),
                    "workflow_error": controller.ledger_error,
                    "prompt_version": config.workflow_prompt_version,
                    "controller_version": config.workflow_controller_version,
                }
                store.event(
                    f"V{self.workflow_version}_TASK_TERMINATED_BEFORE_P0",
                    task_id=task_id,
                    ledger_quality=controller.ledger_quality,
                    error=controller.ledger_error,
                )
                store.write_json(
                    "manifest.json",
                    {
                        "schema_version": f"v{self.workflow_version}-run-1",
                        "final_state": "LEDGER_SCHEMA_FAILED",
                        "workflow_state": "specification_unavailable",
                        "route": "terminated_before_p0",
                        "p0_generated": False,
                        "p1_generated": False,
                        "repair_executed": False,
                        "contract_version": self.workflow_version,
                        "prompt_version": config.workflow_prompt_version,
                        "controller_version": config.workflow_controller_version,
                        "review_protocol": "v5",
                        "review_prompt_version": config.prompt_version,
                        "review_controller_version": config.controller_version,
                        "ledger_quality": controller.ledger_quality,
                        "ledger_warnings": list(controller.ledger_warnings),
                        "workflow_error": controller.ledger_error,
                        "base_commit": base_commit,
                        "baseline_patch_sha256": environment_baseline_sha256,
                        "stage_models": metadata.get("stage_models"),
                    },
                )
                context.metadata = metadata
                self.save_trajectory(
                    self._aggregate_trajectories(instruction, root),
                    context,
                )
                await self.commit_solution(environment)
                return

            ledger_sha256 = sha256_json(ledger)
            write_stage(
                root,
                stage="A",
                task_id=task_id,
                base_commit=base_commit,
                baseline_sha256=environment_baseline_sha256,
                ledger_sha256=ledger_sha256,
                quality=controller.ledger_quality,
                reused_from=(str(fork.source("A")) if fork is not None and "A" in fork.sources else None),
            )
            rendered_solver_prompt = solver_prompt(
                instruction,
                ledger,
                ledger_quality=controller.ledger_quality,
                ledger_warnings=controller.ledger_warnings,
            )
            solver_prompt_sha256 = sha256_bytes(
                rendered_solver_prompt.encode("utf-8")
            )
            if fork is not None and "Solver" in fork.sources:
                p0 = source("Solver").patch("Solver", ledger_sha256=ledger_sha256)
            else:
                solve = await runner.run_role(
                    role=RoleName.SOLVER,
                    prompt=rendered_solver_prompt,
                    worktree=paths.solver,
                    tools=("read", "bash", "edit", "write", "grep", "find", "ls"),
                    timeout_sec=self.workflow_timeout_sec or 10_800,
                    artifact_prefix="v6/solver",
                    invocation_id=f"{specification_stage_id}.solver",
                )
                if solve.exit_code != 0 or solve.timed_out:
                    raise RuntimeError(
                        f"V6 solver failed: {solve.exception or solve.exit_code}"
                    )
                p0 = await workspace.capture_patch(
                    worktree=paths.solver,
                    base_commit=base_commit,
                    baseline_patch=environment_patch,
                )
            store.write_bytes("outputs/v6-p0.patch", p0)
            write_stage(
                root,
                stage="Solver",
                task_id=task_id,
                base_commit=base_commit,
                baseline_sha256=environment_baseline_sha256,
                ledger_sha256=ledger_sha256,
                patch_sha256=sha256_bytes(p0),
                solver_prompt_sha256=solver_prompt_sha256,
                reused_from=(str(fork.source("Solver")) if fork is not None and "Solver" in fork.sources else None),
            )

            regression_worktree = await workspace.prepare_additional_worktree(
                name="regression-review",
                base_commit=base_commit,
            )
            if environment_patch:
                await workspace.apply_patch(
                    worktree=regression_worktree,
                    patch=environment_patch,
                )
            await workspace.apply_patch(worktree=paths.patch_review, patch=p0)
            await workspace.apply_patch(worktree=regression_worktree, patch=p0)
            summary_worktree = paths.summary or paths.patch_review
            if summary_worktree != paths.patch_review:
                await workspace.apply_patch(worktree=summary_worktree, patch=p0)

            review_run_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"v6-review:{task_id}:{base_commit}:{sha256_bytes(p0)}:"
                f"{ledger_sha256}:{solver_prompt_sha256}:"
                f"{environment_baseline_sha256}:"
                f"{self.stage_model_config.digest}:"
                f"{config.workflow_prompt_version}:"
                f"{config.workflow_controller_version}:"
                f"{config.prompt_version}:{config.controller_version}"
                f"{':' + fork.digest if fork is not None else ''}",
            ).hex
            reused_reviews = (
                {
                    role: source(role).review(
                        role,
                        ledger_sha256=ledger_sha256,
                        p0_sha256=sha256_bytes(p0),
                    )
                    for role in ("I", "R") if role in fork.sources
                }
                if fork is not None else {}
            )
            reused_summary = (
                source("S").summary(
                    ledger_sha256=ledger_sha256,
                    p0_sha256=sha256_bytes(p0),
                    review_sha256={
                        role: sha256_json(reused_reviews[role]["report"])
                        for role in ("I", "R")
                    },
                )
                if fork is not None and "S" in fork.sources else None
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
                    run_id=review_run_id,
                    reused_reviews=reused_reviews,
                    reused_summary=reused_summary,
                    export_v6_stages=True,
                )
                final_patch = result.p1_patch
                metadata = {
                    **self.model_spec_for_role("solver").to_metadata(),
                    "stage_models": self.stage_model_config.to_metadata(),
                    "workflow": f"solve-review-repair-v{self.workflow_version}",
                    "workflow_state": "completed",
                    "route": result.route,
                    "p0_sha256": result.p0_sha256,
                    "p1_sha256": sha256_bytes(result.p1_patch),
                    "ledger_sha256": ledger_sha256,
                    "ledger_quality": controller.ledger_quality,
                    "ledger_warnings": list(controller.ledger_warnings),
                    "solver_prompt_sha256": solver_prompt_sha256,
                    "patch_base": (
                        "environment-baseline" if environment_patch else "base-commit"
                    ),
                    "freeze_id": result.freeze_id,
                    "repair_executed": result.route == "repair",
                    "environment_baseline_sha256": environment_baseline_sha256,
                    "environment_baseline": baseline_manifest,
                    "prompt_version": config.workflow_prompt_version,
                    "controller_version": config.workflow_controller_version,
                    "review_prompt_version": config.prompt_version,
                    "review_controller_version": config.controller_version,
                    "fork_plan_sha256": fork.digest if fork is not None else None,
                }
            except Exception as error:
                if fork is not None:
                    raise
                store.event(
                    "V6_REVIEW_SAFE_FALLBACK",
                    error=str(error),
                    task_id=task_id,
                )
                final_patch = p0
                metadata = {
                    **self.model_spec_for_role("solver").to_metadata(),
                    "stage_models": self.stage_model_config.to_metadata(),
                    "workflow": f"solve-review-repair-v{self.workflow_version}",
                    "workflow_state": "safe_fallback",
                    "route": "hold",
                    "workflow_error": self.redact(str(error)),
                    "p0_sha256": sha256_bytes(p0),
                    "p1_sha256": sha256_bytes(p0),
                    "ledger_sha256": ledger_sha256,
                    "ledger_quality": controller.ledger_quality,
                    "ledger_warnings": list(controller.ledger_warnings),
                    "solver_prompt_sha256": solver_prompt_sha256,
                    "patch_base": (
                        "environment-baseline" if environment_patch else "base-commit"
                    ),
                    "freeze_id": None,
                    "repair_executed": None,
                    "environment_baseline_sha256": environment_baseline_sha256,
                    "environment_baseline": baseline_manifest,
                    "prompt_version": config.workflow_prompt_version,
                    "controller_version": config.workflow_controller_version,
                    "review_prompt_version": config.prompt_version,
                    "review_controller_version": config.controller_version,
                    "fork_plan_sha256": fork.digest if fork is not None else None,
                }

            store.write_json(
                "manifest.json",
                {
                    "schema_version": f"v{self.workflow_version}-run-1",
                    "task_id": task_id,
                    "base_commit": base_commit,
                    "final_state": "COMPLETED",
                    "review_protocol": "v5",
                    "workflow_state": metadata["workflow_state"],
                    "route": metadata["route"],
                    "repair_executed": metadata["repair_executed"],
                    "contract_version": self.workflow_version,
                    "prompt_version": metadata["prompt_version"],
                    "controller_version": metadata["controller_version"],
                    "review_prompt_version": metadata["review_prompt_version"],
                    "review_controller_version": metadata[
                        "review_controller_version"
                    ],
                    "ledger_sha256": ledger_sha256,
                    "ledger_quality": controller.ledger_quality,
                    "ledger_warnings": list(controller.ledger_warnings),
                    "solver_prompt_sha256": solver_prompt_sha256,
                    "p0_sha256": metadata["p0_sha256"],
                    "p1_sha256": metadata["p1_sha256"],
                    "patch_base": metadata["patch_base"],
                    "baseline_patch_sha256": environment_baseline_sha256,
                    "freeze_id": metadata["freeze_id"],
                    "environment_baseline": baseline_manifest,
                    "stage_models": metadata.get("stage_models"),
                    "fork_plan_sha256": metadata["fork_plan_sha256"],
                },
            )
            await self._publish_v6_patch(
                workspace=workspace,
                store=store,
                base_commit=base_commit,
                environment_patch=environment_patch,
                baseline_manifest=baseline_manifest,
                final_patch=final_patch,
                verify_exact=fork is not None,
            )
            await self.commit_solution(environment)
            self.save_trajectory(
                self._aggregate_trajectories(instruction, root),
                context,
            )
            context.metadata = metadata
        finally:
            if controller is not None:
                controller.close()
            await workspace.cleanup()

    async def _publish_v6_patch(
        self,
        *,
        workspace: ContainerGitWorkspaceManager,
        store: ArtifactStore,
        base_commit: str,
        environment_patch: bytes,
        baseline_manifest: dict[str, Any] | None,
        final_patch: bytes,
        verify_exact: bool,
    ) -> None:
        discarded_final = await workspace.restore_final_worktree(base_commit=base_commit)
        if discarded_final:
            store.write_bytes("outputs/discarded_final_worktree.patch", discarded_final)
        if environment_patch:
            await workspace.apply_patch(worktree=self.workdir, patch=environment_patch)
        store.event(
            "V6_FINAL_WORKTREE_NORMALIZED",
            discarded_changes=bool(discarded_final),
            discarded_patch_sha256=sha256_bytes(discarded_final),
            baseline_restored=bool(environment_patch),
            baseline_patch_sha256=sha256_bytes(environment_patch),
        )
        await workspace.publish_final_patch(
            base_commit=base_commit,
            patch=final_patch,
            baseline_patch=environment_patch,
            baseline_manifest=baseline_manifest,
        )
        if verify_exact:
            submitted_patch = await workspace.capture_patch(
                worktree=self.workdir,
                base_commit=base_commit,
                baseline_patch=environment_patch,
            )
            store.write_bytes("outputs/submitted.patch", submitted_patch)
            if sha256_bytes(submitted_patch) != sha256_bytes(final_patch):
                raise RuntimeError("V6 submitted patch differs from the selected P1")
