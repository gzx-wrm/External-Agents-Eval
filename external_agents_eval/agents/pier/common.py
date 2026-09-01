from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any

from pier.agents.base import BaseAgent
from pier.environments.base import BaseEnvironment, ExecResult
from pier.models.agent.context import AgentContext
from pier.models.agent.network import NetworkAllowlist
from pier.models.trajectories import Trajectory
from pier.utils.trajectory_metrics import populate_context_from_final_metrics
from pier.utils.trajectory_utils import format_trajectory_json

from external_agents_eval.agents.config import DEFAULT_MODEL


class DeepSWEPierAgent(BaseAgent):
    """Shared Pier contract for external agents evaluated on DeepSWE v1.1."""

    SUPPORTS_ATIF = True
    SUPPORTS_WINDOWS = False
    workdir = "/app"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self.extra_env = dict(extra_env or {})

    @property
    def runtime_model(self) -> str:
        model = self.model_name or DEFAULT_MODEL
        if model.startswith("deepseek/"):
            return model.split("/", 1)[1]
        return model

    def network_allowlist(self) -> NetworkAllowlist:
        return NetworkAllowlist(domains=["api.deepseek.com"])

    def agent_env(self, environment: BaseEnvironment) -> dict[str, str]:
        api_key = self.extra_env.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY was not passed to the Pier agent")
        return environment.agent_process_env({"DEEPSEEK_API_KEY": api_key}) or {}

    def redact(self, text: str | None) -> str:
        value = "" if text is None else str(text)
        secret = self.extra_env.get("DEEPSEEK_API_KEY")
        return value.replace(secret, "[REDACTED]") if secret else value

    def redact_payload(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, list):
            return [self.redact_payload(item) for item in value]
        if isinstance(value, dict):
            return {key: self.redact_payload(item) for key, item in value.items()}
        return value

    async def checked_exec(
        self,
        environment: BaseEnvironment,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        result = await environment.exec(
            command,
            cwd=cwd,
            env=env,
            user=user,
            timeout_sec=timeout_sec,
        )
        if result.return_code != 0:
            output = self.redact(result.stderr or result.stdout or "no output")
            raise RuntimeError(f"Command exited with status {result.return_code}: {output[-2000:]}")
        return result

    async def upload_instruction(self, instruction: str, environment: BaseEnvironment) -> str:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        host_path = self.logs_dir / "instruction.txt"
        host_path.write_text(instruction, encoding="utf-8")
        target = "/tmp/external-agent-instruction.txt"
        await environment.upload_file(host_path, target)
        return target

    async def commit_solution(self, environment: BaseEnvironment) -> None:
        command = """
git config --global --add safe.directory /app
git add -A
if ! git diff --cached --quiet; then
  git -c user.name=external-agents-eval \
      -c user.email=eval@localhost \
      commit -m 'External agent solution'
fi
""".strip()
        await self.checked_exec(environment, command, cwd=self.workdir)

    def save_trajectory(
        self,
        payload: dict[str, Any],
        context: AgentContext,
    ) -> None:
        trajectory = Trajectory.model_validate(payload)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "trajectory.json").write_text(
            format_trajectory_json(trajectory.to_json_dict()) + "\n",
            encoding="utf-8",
        )
        if trajectory.final_metrics is not None:
            populate_context_from_final_metrics(context, trajectory.final_metrics)
        context.n_agent_steps = sum(step.source == "agent" for step in trajectory.steps)

    @staticmethod
    def save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    @abstractmethod
    async def setup(self, environment: BaseEnvironment) -> None: ...

    @abstractmethod
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None: ...
