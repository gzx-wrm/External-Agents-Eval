from __future__ import annotations

import json
import shlex
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
from external_agents_eval.agents.pier.pi_model_config import (
    PiModelSpec,
    PiStageModelConfig,
)


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
        stage_model_config: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self.extra_env = dict(extra_env or {})
        self.stage_model_config = PiStageModelConfig.from_value(
            stage_model_config,
            fallback_model=model_name or DEFAULT_MODEL,
        )

    @property
    def runtime_model(self) -> str:
        return self.model_spec_for_role("solver").model

    @property
    def runtime_provider(self) -> str:
        return self.model_spec_for_role("solver").provider

    def model_spec_for_role(self, role: Any) -> PiModelSpec:
        return self.stage_model_config.for_role(role)

    def network_allowlist(self) -> NetworkAllowlist:
        return NetworkAllowlist(domains=list(self.stage_model_config.domains()))

    def agent_env(self, environment: BaseEnvironment) -> dict[str, str]:
        values: dict[str, str] = {}
        missing: list[str] = []
        for env_key in self.stage_model_config.required_env_keys():
            value = self.extra_env.get(env_key)
            if value:
                values[env_key] = value
            else:
                missing.append(env_key)
        if missing:
            raise RuntimeError(
                "The following provider API keys were not passed to the Pier agent: "
                + ", ".join(missing)
            )
        return environment.agent_process_env(values) or {}

    def redact(self, text: str | None) -> str:
        value = "" if text is None else str(text)
        for env_key in self.stage_model_config.required_env_keys():
            secret = self.extra_env.get(env_key)
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value

    async def prepare_pi_config(
        self,
        environment: BaseEnvironment,
        model_spec: PiModelSpec,
        *,
        config_key: str,
        config_dir: str = "/external_eval/pi_config",
    ) -> str:
        """Write a provider override without ever writing API credentials."""
        models_json = self.stage_model_config.models_json(model_spec)
        if models_json is None:
            return config_dir
        if not config_key.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Unsafe Pi model config key: {config_key}")
        target_dir = (
            "/external_eval/pi_model_configs/"
            f"{config_key}-{self.stage_model_config.digest[:16]}"
        )
        payload = json.dumps(models_json, ensure_ascii=False, separators=(",", ":"))
        settings = json.dumps(
            {"defaultProjectTrust": "never", "telemetry": False},
            separators=(",", ":"),
        )
        await self.checked_exec(
            environment,
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"printf '%s' {shlex.quote(payload)} > {shlex.quote(target_dir + '/models.json')} && "
            f"printf '%s' {shlex.quote(settings)} > "
            f"{shlex.quote(target_dir + '/settings.json')}"
        )
        return target_dir

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
