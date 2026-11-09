from __future__ import annotations

import json
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
from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    load_frozen_few_shots,
    load_reviewer_experiment_manifest,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore
from external_agents_eval.workflows.solve_review_repair.config import WorkflowConfig
from external_agents_eval.workflows.solve_review_repair.controller import (
    SolveReviewRepairController,
)
from external_agents_eval.workflows.solve_review_repair.models import TaskInput


def resolve_frozen_task_id(
    session_task_id: str,
    frozen_task_ids: tuple[str, ...],
) -> str:
    """Recover the full task ID from Pier's normalized trial-name prefix."""
    if session_task_id in frozen_task_ids:
        return session_task_id
    matches = tuple(
        task_id for task_id in frozen_task_ids if task_id[:32].rstrip("_-") == session_task_id
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Task {session_task_id} is not in the frozen reviewer task set")
    raise ValueError(f"Task prefix {session_task_id} is ambiguous in the frozen reviewer task set")


def issue_title_from_instruction(instruction: str, task_id: str) -> str:
    """Use a real short heading, otherwise fall back to the stable task ID."""
    first_line = next((line.strip() for line in instruction.splitlines() if line.strip()), "")
    candidate = first_line.lstrip("#").strip()
    return candidate if candidate and len(candidate) <= 500 else task_id


class PiSolveReviewRepairPierAgent(PiPierAgent):
    """Pier-facing composite Agent; Pier sees only the final committed P1."""

    def __init__(
        self,
        *args: Any,
        p0_manifest: str | None = None,
        few_shot_dir: str | None = None,
        few_shot_digest: str | None = None,
        review_arm: str = "E0",
        review_contract_version: str | None = None,
        review_prompt_version: str | None = None,
        workflow_timeout_sec: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.p0_manifest_path = Path(p0_manifest).expanduser().resolve() if p0_manifest else None
        self.few_shot_dir = Path(few_shot_dir).expanduser().resolve() if few_shot_dir else None
        self.few_shot_digest = few_shot_digest
        self.review_arm = review_arm
        # Pier's ``--agent-kwarg`` parser coerces unquoted numeric-looking
        # values (for example ``review_contract_version=3.0``) to ``float``.
        # Contract versions are semantic string identifiers, so normalize the
        # value at the agent boundary before comparing it with the supported
        # contract set.  This keeps the CLI command human-readable while
        # preventing a spurious "arm incompatible" failure in real trials.
        self.review_contract_version = (
            str(review_contract_version) if review_contract_version is not None else None
        )
        self.review_prompt_version = (
            str(review_prompt_version) if review_prompt_version is not None else None
        )
        self.workflow_timeout_sec = workflow_timeout_sec

    @staticmethod
    def name() -> str:
        return "pi-solve-review-repair"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        task_id = environment.session_id.rsplit("__", 1)[0]
        head = await self.checked_exec(environment, "git rev-parse HEAD", cwd=self.workdir)
        base_commit = (head.stdout or "").strip()
        existing_p0 = None
        few_shots = load_frozen_few_shots(self.few_shot_dir) if self.few_shot_dir else None
        if few_shots is None:
            raise ValueError("The review workflow requires the frozen development few-shots")
        if self.few_shot_digest is None or few_shots.digest != self.few_shot_digest:
            raise ValueError("Frozen development few-shot digest does not match the batch run")
        if self.review_arm not in {"E0", "E1", "E2"}:
            raise ValueError(f"Unsupported reviewer experiment arm: {self.review_arm}")
        contract_version = self.review_contract_version or (
            "1.0" if self.review_arm == "E0" else "2.0"
        )
        allowed_contracts = {"1.0"} if self.review_arm == "E0" else {"2.0", "3.0"}
        if contract_version not in allowed_contracts:
            raise ValueError(
                f"Experiment arm {self.review_arm} is incompatible with contract {contract_version}"
            )
        if self.review_prompt_version is not None and self.review_prompt_version != "4.0":
            raise ValueError(f"Unsupported reviewer prompt version: {self.review_prompt_version}")
        if self.review_prompt_version == "4.0" and (
            contract_version != "3.0" or self.review_arm not in {"E1", "E2"}
        ):
            raise ValueError(
                "Reviewer prompt v4.0 requires contract v3.0 and experiment arm E1 or E2"
            )
        if few_shots.contract_version != contract_version:
            raise ValueError(
                f"Few-shot contract {few_shots.contract_version} does not match "
                f"experiment arm {self.review_arm}"
            )
        if few_shots.prompt_version != self.review_prompt_version:
            raise ValueError(
                "Few-shot prompt version does not match the selected reviewer prompt version"
            )
        prompt_version = self.review_prompt_version or (
            "1.3" if contract_version == "1.0" else "2.3" if contract_version == "2.0" else "3.0"
        )
        if self.p0_manifest_path is not None:
            manifest = load_reviewer_experiment_manifest(self.p0_manifest_path)
            task_id = resolve_frozen_task_id(task_id, manifest.task_ids)
            existing_p0 = manifest.patches[task_id]

        artifact_root = self.logs_dir.parent / "review_repair"
        store = ArtifactStore(
            artifact_root,
            secrets=tuple(
                self.extra_env.get(key, "")
                for key in self.stage_model_config.required_env_keys()
            ),
        )
        role_env = self.agent_env(environment)
        role_env.update(
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
        role_runner = PiRoleRunner(
            environment=environment,
            logs_root=artifact_root,
            model=self.runtime_model,
            provider=self.runtime_provider,
            stage_model_config=self.stage_model_config,
            agent_version=self.version(),
            env=role_env,
            redact=self.redact,
        )
        workspace = ContainerGitWorkspaceManager(
            environment=environment,
            transfer_dir=artifact_root / ".transfer",
            repository=self.workdir,
            redact=self.redact,
        )
        controller = SolveReviewRepairController(
            role_runner=role_runner,
            workspace=workspace,
            artifacts=store,
            config=WorkflowConfig(
                solver_timeout_sec=self.workflow_timeout_sec or 10_800,
                repair_timeout_sec=self.workflow_timeout_sec or 10_800,
                stage_a_few_shots=few_shots.stage_a,
                stage_b_few_shots=few_shots.stage_b,
                repair_few_shots=few_shots.repair,
                few_shot_metadata=(few_shots.root / "provenance.json",),
                contract_version=contract_version,
                max_obligations=64 if contract_version == "3.0" else 32,
                experiment_arm=self.review_arm,
                prompt_version=prompt_version,
                controller_version=(
                    "2.0.0"
                    if contract_version == "1.0"
                    else "2.0.1"
                    if contract_version == "2.0"
                    else "3.0.1"
                ),
                model=self.model_spec_for_role("solver").model,
                agent_version=self.version(),
                resume=True,
            ),
        )
        result = await controller.run(
            TaskInput(
                task_id=task_id,
                issue_title=issue_title_from_instruction(instruction, task_id),
                issue_body=instruction,
                repository_name=task_id.split("-", 1)[0],
                base_commit=base_commit,
            ),
            existing_p0=existing_p0,
        )
        await self.commit_solution(environment)
        trajectory = self._aggregate_trajectories(instruction, artifact_root)
        self.save_trajectory(trajectory, context)
        context.metadata = {
            **self.model_spec_for_role("solver").to_metadata(),
            "stage_models": self.stage_model_config.to_metadata(),
            "workflow": f"solve-review-repair-once-v{contract_version.split('.')[0]}",
            "experiment_arm": self.review_arm,
            "contract_version": contract_version,
            "prompt_version": prompt_version,
            "workflow_state": result.state.value,
            "reviewer_verdict": result.verdict,
            "repair_executed": result.repair_executed,
            "p0_sha256": result.p0_sha256,
            "p1_sha256": result.p1_sha256,
        }

    def _aggregate_trajectories(self, instruction: str, root: Path) -> dict[str, Any]:
        steps: list[dict[str, Any]] = [{"step_id": 1, "source": "user", "message": instruction}]
        totals = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
        }
        seen_metric = {key: False for key in totals}
        for path in sorted(root.glob("**/trajectory.json")):
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
        final_metrics.update({key: value for key, value in totals.items() if seen_metric[key]})
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
